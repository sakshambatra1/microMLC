from typing import Tuple, List

def make_next_token_pair(tokenizer, text:str, max_len:int) -> Tuple[List[int], List[int]]:
    ids = tokenizer.encode(text, add_special_tokens=False)

    if len(ids) < 2: 
        ids = ids + [tokenizer.eos_token_id]
    input_ids = ids[:-1]
    tgt_ids = ids[1:]


    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [tokenizer.eos_token_id] * (max_len - len(input_ids))


    if len(tgt_ids) > max_len:
        tgt_ids = tgt_ids[:max_len]
    else:
        tgt_ids = tgt_ids + [tokenizer.eos_token_id] * (max_len - len(tgt_ids))

    return input_ids, tgt_ids
