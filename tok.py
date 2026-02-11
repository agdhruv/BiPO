"""
This file includes code adapted from https://github.com/nrimsky/CAA/blob/main/utils/tokenize.py
"""
from typing import List, Tuple, Optional


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def tokenize(
    tokenizer,
    system_prompt: str,
    model_name: str,
    conversation: List[Tuple[str, Optional[str]]],
    no_final_eos=False,
) -> List[int]:
    """
    tokenizer: a HuggingFace tokenizer
    system_prompt: the system prompt to use for the conversation
    model_name: the name of the model
    conversation: a list of tuples of the form (user_input, model_output) - model_output can be None if there is no response yet
    no_final_eos: if True, the final </s> token will not be added to the end of the conversation when there is a model response

    Returns: a list of tokens
    """
    # Use native chat template for Llama-3
    if model_name == 'llama-3':
        messages = [{"role": "system", "content": system_prompt}]
        for user_input, model_output in conversation:
            messages.append({"role": "user", "content": user_input})
            if model_output is not None:
                messages.append({"role": "assistant", "content": model_output})
        
        # If the last message has a model output, we add generation prompt only if there's no output
        add_gen_prompt = conversation[-1][1] is None if conversation else True
        tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_gen_prompt,
            tokenize=True
        )
        # Remove trailing EOS if no_final_eos and there was a model response
        if no_final_eos and not add_gen_prompt and tokens[-1] == tokenizer.eos_token_id:
            tokens = tokens[:-1]
        return tokens
    
    def _instruction_response_to_tokens(
        instruction, model_output=None, is_first_message=False, no_eos=False
    ):
        if is_first_message:
            if model_name == 'llama-2':
                dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
            elif model_name == 'mistral':
                dialog_content = system_prompt + '\n' + instruction.strip()
            else:
                raise SystemExit("Unsupported model name: ", model_name)
        else:
            dialog_content = instruction.strip()
        if model_output is not None:
            if no_eos:
                return tokenizer.encode(
                    f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
                )

            return tokenizer.encode(
                f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()} {tokenizer.eos_token}"
            )
        else:
            return tokenizer.encode(f"{B_INST} {dialog_content.strip()} {E_INST}")

    tokens = []
    for i, (user_input, model_output) in enumerate(conversation):
        tokens += _instruction_response_to_tokens(
            user_input,
            model_output,
            i == 0,
            no_final_eos and (i == len(conversation) - 1),
        )
    return tokens
