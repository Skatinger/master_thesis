from datasets import load_dataset
from custom.splitter import Splitter
from transformers import LongformerTokenizer


# helper function to split the examples into chunks around the masks
# and save as a new dataset, with just sentences with a single mask token and the belonging
# prediction.
# already processed version available on huggingface on rcds/wikipedia-for-fill-mask
# this script only creates a jsonl file. Before uploading to huggingface, compress it to xz
# with `xz -T4 -9e for_mask_prediction.jsonl`.

dataset = load_dataset("rcds/wikipedia-persons-masked", split='train')
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")


def split_examples(examples):
    all_text_chunks = []
    all_mask_chunks = []
    # split the text around the masks
    for text, masks in zip(examples['masked_text_original'], examples['masked_entities_original']):
        text_chunks = [*Splitter.split_by_max_tokens(text, tokenizer, max_tokens=4096)]
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

    return {"text_chunks": all_text_chunks, "masks": all_mask_chunks}


new_dataset = dataset.map(split_examples, batched=True, batch_size=128, remove_columns=dataset.column_names, num_proc=8)
new_dataset.to_json('for_mask_prediction_original_4096.jsonl')
