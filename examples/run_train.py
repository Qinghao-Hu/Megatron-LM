import os
import time
import argparse
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path

from megatron.core import parallel_state
from megatron.core import dist_checkpointing
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.datasets.utils import compile_helpers 
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
from megatron.training.utils import (get_batch_on_this_cp_rank, get_batch_on_this_tp_rank, print_rank_0)


class _NullTokenizer(MegatronTokenizer):
    def __init__(self, vocab_size):
        super().__init__(None, vocab_size=vocab_size)
        self._vocab_size_without_eod = int(vocab_size)
        self._eod_id = self._vocab_size_without_eod

    def tokenize(self, text):
        return [int(x) for x in text.split(' ')]

    def detokenize(self, ids):
        text = [str(x) for x in ids]
        return ' '.join(text)

    @property
    def vocab_size(self):
        return self._vocab_size_without_eod + 1

    @property
    def vocab(self):
        raise NotImplementedError

    @property
    def inv_vocab(self):
        raise NotImplementedError

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self._eod_id

    @property
    def additional_special_tokens_ids(self):
        return None


def initialize_distributed(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_model_parallel_size=1):
    parallel_state.destroy_model_parallel()

    # Torch setup for distributed training
    rank = int(os.environ['LOCAL_RANK'])
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, context_parallel_size=context_model_parallel_size)

def model_provider(args):
    """Build the model."""

    # LLaMA-2 7B
    print("Note Current use layer=2")
    transformer_config = TransformerConfig(
        num_layers=2, 
        hidden_size=4096, 
        num_attention_heads=32, 
        ffn_hidden_size = 11008,
        activation_func = torch.nn.functional.silu,
        add_bias_linear = False,
        bias_activation_fusion = False,
        gated_linear_unit = True,
        apply_query_key_layer_scaling = True,
        layernorm_zero_centered_gamma = False,
        bias_dropout_fusion = False,
        apply_rope_fusion = False,
        use_cpu_initialization=True, 
        pipeline_dtype=torch.float32
    )

    gpt_model = GPTModel(
        config=transformer_config, 
        transformer_layer_spec=get_gpt_layer_local_spec(), 
        vocab_size=32000, 
        max_sequence_length=args.seq_length,
        position_embedding_type="rope",
        rotary_percent = 0.5
    )

    return gpt_model

def get_train_data_iterator():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    config = GPTDatasetConfig(
        random_seed=0,
        sequence_length=args.seq_length,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=_NullTokenizer(vocab_size=args.seq_length),
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, config
    ).build()

    train_dataloader = DataLoader(datasets[0], batch_size=1, shuffle=True)

    train_iterator = iter(train_dataloader)

    return train_iterator


def forward_step_func(data_iterator, model):

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        # If you have data parallel reduce loss across data parallel groups.
        # If pipeline parallel, loss computation is done only in last stage.

        return loss, {'lm loss': loss}

    data = next(data_iterator)
    # data = get_batch(data_iterator)

    # Apply SP Sampling
    # NOTE: This is a dummy implementation
    print('Data:', data['tokens'].shape)
    data = {key: value[:, :value.size(1) // CP_SIZE].to(device) for key, value in data.items()}

    tokens = data['tokens'].to(device)
    attention_mask = data['attention_mask'].to(device)
    position_ids = data['position_ids'].to(device)
    labels = data['labels'].to(device)
    loss_mask = data['loss_mask'].to(device)

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)

def save_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix='')
    dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)

def load_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict=gpt_model.sharded_state_dict(prefix='')
    checkpoint = dist_checkpointing.load(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)
    gpt_model.load_state_dict(checkpoint)
    return gpt_model

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--seed", type=int, default=123)
    args.add_argument("--max_train_steps", type=int, default=10)
    # args.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    args.add_argument("--dataset", type=str, default="dummy")
    args.add_argument("--seq_length", type=int, default=2048)
    args.add_argument(
        "--tensor_parallel_size", type=int, default=1
    )
    args.add_argument(
        "--pipeline_parallel_size", type=int, default=1
    )
    args.add_argument(
        "--context_parallel_size", type=int, default=1, help="i.e., sequence parallel"
    )
    args = args.parse_args()

    global CP_SIZE
    CP_SIZE=args.context_parallel_size

    initialize_distributed(tensor_model_parallel_size=args.tensor_parallel_size, pipeline_model_parallel_size=args.pipeline_parallel_size, context_model_parallel_size=args.context_parallel_size)
    model_parallel_cuda_manual_seed(args.seed)

    gpt_model = model_provider(args)
    device = torch.device("cuda")
    gpt_model.to(device)

    optim = Adam(gpt_model.parameters())

    train_iterator = get_train_data_iterator()

    forward_backward_func = get_forward_backward_func()

    # Running the model for 5 iterations
    for step in range(args.max_train_steps):
        step_start_time = time.time()
        optim.zero_grad()

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=gpt_model,
            # cp_size= args.context_parallel_size,
            num_microbatches=1,
            seq_length=args.seq_length//args.context_parallel_size,
            micro_batch_size=1,
            decoder_seq_length=args.seq_length//args.context_parallel_size,
            forward_only=False)

        optim.step()

        print(f"Step {step}, Step Time {time.time()-step_start_time}, Loss:  {losses_reduced}")

    # # Saving the model
    # ckpt_path = os.getcwd() + '/ckpt'
    # Path(ckpt_path).mkdir(exist_ok=True)
    # save_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_path)

    # # Loading the model
    # gpt_model = load_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_path)
    # gpt_model.to(device)
    # print('Successfully loaded the model')
