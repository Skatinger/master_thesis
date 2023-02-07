from datasets import load_dataset
from custom.splitter import Splitter
from transformers import LongformerTokenizer
from transformers import AutoTokenizer


# helper function to split the examples into chunks around the masks
# and save as a new dataset, with just sentences with a single mask token and the belonging
# prediction.
# already processed version available on huggingface on rcds/wikipedia-for-fill-mask
# this script only creates a jsonl file. Before uploading to huggingface, compress it to xz
# with `xz -T4 -9e for_mask_prediction.jsonl`.

dataset = load_dataset("rcds/wikipedia-persons-masked", split='train')


def split_original_4096_examples(examples):
    all_text_chunks = []
    all_mask_chunks = []
    # split the text around the masks
    for text, masks in zip(examples["masked_text_original"], examples["masked_entities_original"]):
        text_chunks = [*Splitter.split_by_max_tokens(text, tokenizer, max_tokens=tokenizer.model_max_length)]
        # for each text_chunk, get the number of masks in it, and get the corresponding masks
        last_chunk_mask_index = 0
        for chunk in text_chunks:
            # get the number of masks in the chunk
            num_masks = chunk.count('<mask>')
            # get the corresponding masks
            chunk_masks = masks[last_chunk_mask_index:(num_masks + last_chunk_mask_index)]
            # update the last_chunk_mask_index
            last_chunk_mask_index += num_masks
            # add the masks to the mask_chunks
            all_mask_chunks += [chunk_masks]
        all_text_chunks += text_chunks
        types = ["original"] * len(all_text_chunks)
        sizes = [4096] * len(all_text_chunks)

    return {"type": types, "size": sizes, "texts": all_text_chunks, "masks": all_mask_chunks}


def split_paraphrased_4096_examples(examples):
    all_text_chunks = []
    all_mask_chunks = []
    # split the text around the masks
    for text, masks in zip(examples["masked_text_paraphrased"], examples["masked_entities_paraphrased"]):
        text_chunks = [*Splitter.split_by_max_tokens(text, tokenizer, max_tokens=tokenizer.model_max_length)]
        # for each text_chunk, get the number of masks in it, and get the corresponding masks
        last_chunk_mask_index = 0
        for chunk in text_chunks:
            # get the number of masks in the chunk
            num_masks = chunk.count('<mask>')
            # get the corresponding masks
            chunk_masks = masks[last_chunk_mask_index:(num_masks + last_chunk_mask_index)]
            # update the last_chunk_mask_index
            last_chunk_mask_index += num_masks
            # add the masks to the mask_chunks
            all_mask_chunks += [chunk_masks]
        all_text_chunks += text_chunks
        types = ["paraphrased"] * len(all_text_chunks)
        sizes = [4096] * len(all_text_chunks)

    return {"type": types, "size": sizes, "texts": all_text_chunks, "masks": all_mask_chunks}


def split_original_512_examples(examples):
    all_text_chunks = []
    all_mask_chunks = []
    # split the text around the masks
    for text, masks in zip(examples["masked_text_original"], examples["masked_entities_original"]):
        text_chunks = [*Splitter.split_by_max_tokens(text, tokenizer, max_tokens=tokenizer.model_max_length)]
        # for each text_chunk, get the number of masks in it, and get the corresponding masks
        last_chunk_mask_index = 0
        for chunk in text_chunks:
            # get the number of masks in the chunk
            num_masks = chunk.count('<mask>')
            # get the corresponding masks
            chunk_masks = masks[last_chunk_mask_index:(num_masks + last_chunk_mask_index)]
            # update the last_chunk_mask_index
            last_chunk_mask_index += num_masks
            # add the masks to the mask_chunks
            all_mask_chunks += [chunk_masks]
        all_text_chunks += text_chunks
        types = ["original"] * len(all_text_chunks)
        sizes = [512] * len(all_text_chunks)

    return {"type": types, "size": sizes, "texts": all_text_chunks, "masks": all_mask_chunks}


def split_paraphrased_512_examples(examples):
    all_text_chunks = []
    all_mask_chunks = []
    # split the text around the masks
    for text, masks in zip(examples["masked_text_paraphrased"], examples["masked_entities_paraphrased"]):
        text_chunks = [*Splitter.split_by_max_tokens(text, tokenizer, max_tokens=tokenizer.model_max_length)]
        # for each text_chunk, get the number of masks in it, and get the corresponding masks
        last_chunk_mask_index = 0
        for chunk in text_chunks:
            # get the number of masks in the chunk
            num_masks = chunk.count('<mask>')
            # get the corresponding masks
            chunk_masks = masks[last_chunk_mask_index:(num_masks + last_chunk_mask_index)]
            # update the last_chunk_mask_index
            last_chunk_mask_index += num_masks
            # add the masks to the mask_chunks
            all_mask_chunks += [chunk_masks]
        all_text_chunks += text_chunks
        types = ["paraphrased"] * len(all_text_chunks)
        sizes = [512] * len(all_text_chunks)

    return {"type": types, "size": sizes, "texts": all_text_chunks, "masks": all_mask_chunks}


tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
original_4096 = dataset.map(split_original_4096_examples, batched=True, batch_size=128, remove_columns=dataset.column_names, num_proc=6)
original_4096 = original_4096.filter(lambda row: '<mask>' in row['texts'])
original_4096.to_json('original_4096.jsonl')
paraphrased_4096 = dataset.map(split_paraphrased_4096_examples, batched=True, batch_size=128, remove_columns=dataset.column_names, num_proc=6)
paraphrased_4096 = paraphrased_4096.filter(lambda row: '<mask>' in row['texts'])
paraphrased_4096.to_json('paraphrased_4096.jsonl')
# overwrite tokenizer to use xlm-roberta-large for the smaller chunks
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
original_512 = dataset.map(split_original_512_examples, batched=True, batch_size=128, remove_columns=dataset.column_names, num_proc=6)
original_512 = original_512.filter(lambda row: '<mask>' in row['texts'])
original_512.to_json('original_512.jsonl')
paraphrased_512 = dataset.map(split_original_512_examples, batched=True, batch_size=128, remove_columns=dataset.column_names, num_proc=6)
paraphrased_512 = paraphrased_512.filter(lambda row: '<mask>' in row['texts'])
paraphrased_512.to_json('paraphrased_512.jsonl')
