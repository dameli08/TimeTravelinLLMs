from openai import OpenAI
import re


class OpenAIClient:
    def __init__(self, base_url=None):
        if base_url:
            self.client = OpenAI(api_key="EMPTY", base_url=base_url)
        else:
            self.client = OpenAI()

    def get_text(
        self,
        text,
        model,
        max_tokens=500,
        temperature=0.0,
        top_p=1.00,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        system_message=None,
        thinking_mode=False,
        thinking_budget=1000,
    ):
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": text})

        if thinking_mode:
            chat_tmpl_kwargs = {"enable_thinking": True}
            if thinking_budget and thinking_budget > 0:
                chat_tmpl_kwargs["thinking_budget"] = thinking_budget
            extra_body = {"chat_template_kwargs": chat_tmpl_kwargs}
            # Inject </think> as an assistant prefix.  The Qwen3 chat template
            # prepends <think> to every assistant turn, so the model receives
            # <think></think> — an empty thinking block — and skips straight to
            # generating the answer, avoiding the endless thinking loop that
            # occurs when the model is asked to do exact recall in thinking mode.
            messages.append({"role": "assistant", "content": "</think>\n\n"})
        else:
            extra_body = {}

        # Try making the API call
        try:
            response = self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                messages=messages,
                extra_body=extra_body if extra_body else None,
            )
        except Exception as e:
            raise Exception(f"Failed to create completion with OpenAI API: {str(e)}")

        # Check if the response has valid data
        if response.choices and len(response.choices) > 0:
            first_choice = response.choices[0]

            if first_choice.message and first_choice.message.content:
                content = str(first_choice.message.content)
                if thinking_mode:
                    if '</think>' in content:
                        # Server echoed our injected prefix; strip up to </think>.
                        content = content[content.find('</think>') + len('</think>'):].strip()
                    elif first_choice.finish_reason == 'length':
                        # Token limit hit while still inside the thinking block —
                        # assistant-prefill not supported or not enforced by the server.
                        # No usable answer was produced.
                        content = ""
                    # else: finish_reason='stop', no </think> in content.
                    # The server stripped our injected prefix before returning;
                    # content is already the clean answer — return it as-is.
                return content
            else:
                raise Exception(
                    "Response from OpenAI API does not "
                    "contain 'message' or 'content'."
                )
        else:
            raise Exception(
                "Response from OpenAI API does not contain "
                "'choices' or choices list is empty."
            )
