from openai import OpenAI


THINKING_MAX_TOKENS = 12000


class OpenAIClient:
    def __init__(self, base_url=None):
        self.is_local_endpoint = base_url is not None
        if base_url:
            self.client = OpenAI(api_key="EMPTY", base_url=base_url)
        else:
            self.client = OpenAI()

    def get_text(
        self,
        text,
        model,
        max_tokens=None,
        temperature=None,
        top_p=None,
        top_k=None,
        min_p=None,
        frequency_penalty=None,
        presence_penalty=None,
        repetition_penalty=None,
        system_message=None,
        thinking_mode=False,
        thinking_budget=THINKING_MAX_TOKENS,
        reasoning_effort=None,
    ):
        original_temperature = temperature
        original_top_p = top_p
        original_frequency_penalty = frequency_penalty
        original_presence_penalty = presence_penalty

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": text})

        extra_body = {}

        if thinking_mode and self.is_local_endpoint:
            chat_template_kwargs = {"enable_thinking": True}
            if thinking_budget and thinking_budget > 0:
                chat_template_kwargs["thinking_budget"] = thinking_budget
            extra_body["chat_template_kwargs"] = chat_template_kwargs

            temperature = 1.0 if temperature is None else temperature
            top_p = 0.95 if top_p is None else top_p
            top_k = 20 if top_k is None else top_k
            min_p = 0.0 if min_p is None else min_p
            presence_penalty = 1.5 if presence_penalty is None else presence_penalty
            repetition_penalty = 1.0 if repetition_penalty is None else repetition_penalty

            if max_tokens is None:
                # Keep thinking-mode generation at 12k tokens. This matches the
                # observed Qwen thinking behavior for these contamination runs.
                max_tokens = THINKING_MAX_TOKENS
        else:
            # Explicitly disable thinking — Qwen3 models default to thinking-enabled
            # in their chat template, so we must pass False to suppress it.
            if self.is_local_endpoint:
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}

            temperature = 0.0 if temperature is None else temperature
            top_p = 1.0 if top_p is None else top_p
            presence_penalty = 0.0 if presence_penalty is None else presence_penalty
            if max_tokens is None:
                max_tokens = 500

        frequency_penalty = 0.0 if frequency_penalty is None else frequency_penalty
        if self.is_local_endpoint and top_k is not None:
            extra_body["top_k"] = top_k
        if self.is_local_endpoint and min_p is not None:
            extra_body["min_p"] = min_p
        if self.is_local_endpoint and repetition_penalty is not None:
            extra_body["repetition_penalty"] = repetition_penalty

        # Try making the API call
        try:
            request_kwargs = {
                "model": model,
                "messages": messages,
            }

            if self.is_local_endpoint:
                request_kwargs.update(
                    {
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "frequency_penalty": frequency_penalty,
                        "presence_penalty": presence_penalty,
                    }
                )
                if extra_body:
                    request_kwargs["extra_body"] = extra_body
            else:
                if str(model).startswith("gpt-5"):
                    request_kwargs["max_completion_tokens"] = max_tokens
                    if reasoning_effort is not None:
                        request_kwargs["reasoning_effort"] = reasoning_effort
                else:
                    request_kwargs["max_tokens"] = max_tokens

                if original_temperature is not None:
                    request_kwargs["temperature"] = original_temperature
                if original_top_p is not None:
                    request_kwargs["top_p"] = original_top_p
                if original_frequency_penalty is not None:
                    request_kwargs["frequency_penalty"] = original_frequency_penalty
                if original_presence_penalty is not None:
                    request_kwargs["presence_penalty"] = original_presence_penalty

            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as e:
            raise Exception(f"Failed to create completion with OpenAI API: {str(e)}")

        # Check if the response has valid data
        if response.choices and len(response.choices) > 0:
            first_choice = response.choices[0]

            if first_choice.message and first_choice.message.content:
                content = str(first_choice.message.content)
                if thinking_mode:
                    if '</think>' in content:
                        # Strip everything up to and including </think>, keep only the answer.
                        content = content[content.find('</think>') + len('</think>'):].strip()
                    else:
                        # Model hit token limit mid-thinking — no answer produced.
                        content = ""
                return content
            else:
                finish_reason = getattr(first_choice, "finish_reason", None)
                usage = getattr(response, "usage", None)
                raise Exception(
                    "Response from OpenAI API does not "
                    f"contain 'message' or 'content'. finish_reason={finish_reason}, "
                    f"usage={usage}"
                )
        else:
            raise Exception(
                "Response from OpenAI API does not contain "
                "'choices' or choices list is empty."
            )
