import argparse


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument(
            "--filepath",
            required=True,
            type=str,
            help="The filepath to the CSV file containing the original dataset instances.",
        )
        self.parser.add_argument(
            "--task",
            required=True,
            type=str,
            choices=["cls", "nli", "sum", "xsum", "qa"],
            help="The task corresponding to the dataset. "
            "For NLI and classification tasks, label column should be specified. "
            "(Choices: %(choices)s)",
        )
        self.parser.add_argument(
            "--dataset",
            required=True,
            type=str,
            help="Dataset name.",
        )
        self.parser.add_argument(
            "--split",
            required=True,
            type=str,
            choices=["train", "test", "validation"],
            help="Dataset partition. (Choices: %(choices)s)",
        )
        self.parser.add_argument(
            "--model",
            required=True,
            type=str,
            help="Model name to be evaluated for contamination. "
            "Select an OpenAI model snapshot, such as a version "
            "of GPT-4 or GPT-3.5.",
        )
        self.parser.add_argument(
            "--text_column",
            required=True,
            nargs="+",
            type=str,
            help="Column name for where the replication should be "
            "performed on. For NLI task, provide column name for "
            "sentence1/premise and sentence2/hypothesis, respetively, "
            "separated by a single space.",
        )
        self.parser.add_argument(
            "--label_column",
            type=str,
            default=None,
            help="Column name for labels corresponding to the dataset instances "
            "if the task comes with label.",
        )
        self.parser.add_argument(
            "--should_split_text",
            action="store_true",
            help="Use it to split text randomly. For pre-split text, "
            "ensure it is in two columns named 'first_piece' "
            "and 'second_piece' for single-instance datasets.",
        )
        self.parser.add_argument(
            "--min_p",
            type=float,
            default=40.0,
            help="Specify the minimum percentage for the range "
            "[min_p, max_p], that will randomly determine the length of the "
            "first piece of text.",
        )
        self.parser.add_argument(
            "--max_p",
            type=float,
            default=70.0,
            help="Specify the maximum percentage for the range "
            "[min_p, max_p], that will randomly determine the length of the "
            "first piece of text.",
        )
        self.parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Set the seed value upon which random text splits will be based.",
        )
        self.parser.add_argument(
            "--base_url",
            type=str,
            default=None,
            help="Base URL for a local OpenAI-compatible API (e.g., vLLM). "
                 "If not set, uses the OpenAI API with OPENAI_API_KEY.",
        )
        self.parser.add_argument(
            "--sleep_time",
            type=float,
            default=3.0,
            help="Seconds to sleep between API calls. Set to 0 for local endpoints.",
        )
        self.parser.add_argument(
            "--sample_size",
            type=int,
            default=None,
            help="If set, randomly sample exactly this many rows from the dataset. "
            "A fresh random sample is drawn each run.",
        )
        self.parser.add_argument(
            "--system_message",
            type=str,
            default=None,
            help="Optional system message prepended to every API call. Useful for "
            "restricting model output (e.g. disabling thinking mode with /no_think).",
        )
        self.parser.add_argument(
            "--thinking_mode",
            action="store_true",
            help="Enable model thinking/reasoning mode. Passes enable_thinking=True to vLLM "
            "and strips <think>...</think> blocks from the response before storing.",
        )
        self.parser.add_argument(
            "--thinking_budget",
            type=int,
            default=12000,
            help="Maximum number of tokens the model may use for thinking (reasoning) when "
            "--thinking_mode is set. Passed as thinking_budget in chat_template_kwargs. "
            "Prevents the model from exhausting max_tokens on thinking alone. "
            "Set to 0 to disable the budget (unlimited thinking).",
        )
        self.parser.add_argument(
            "--max_tokens",
            type=int,
            default=None,
            help="Maximum number of tokens to generate. In thinking mode the default is 12000. "
            "In normal mode the default is 500.",
        )
        self.parser.add_argument(
            "--temperature",
            type=float,
            default=None,
            help="Sampling temperature. Defaults to 1.0 in thinking mode and 0.0 otherwise.",
        )
        self.parser.add_argument(
            "--top_p",
            type=float,
            default=None,
            help="Nucleus sampling probability. Defaults to 0.95 in thinking mode and 1.0 otherwise.",
        )
        self.parser.add_argument(
            "--top_k",
            type=int,
            default=None,
            help="Top-k sampling value for OpenAI-compatible local endpoints. Defaults to 20 in thinking mode.",
        )
        self.parser.add_argument(
            "--sampling_min_p",
            type=float,
            default=None,
            help="Minimum probability sampling value for OpenAI-compatible local endpoints. Defaults to 0.0 in thinking mode.",
        )
        self.parser.add_argument(
            "--frequency_penalty",
            type=float,
            default=None,
            help="Frequency penalty. Defaults to 0.0.",
        )
        self.parser.add_argument(
            "--presence_penalty",
            type=float,
            default=None,
            help="Presence penalty. Defaults to 1.5 in thinking mode and 0.0 otherwise.",
        )
        self.parser.add_argument(
            "--repetition_penalty",
            type=float,
            default=None,
            help="Repetition penalty for OpenAI-compatible local endpoints. Defaults to 1.0 in thinking mode.",
        )
        self.parser.add_argument(
            "--bleurt_eval",
            action="store_true",
            help="If the evaluation should be performed based on the BLEURT score.",
        )
        self.parser.add_argument(
            "--rouge_eval",
            action="store_true",
            help="If the evaluation should be performed based on the ROUGE-L score.",
        )
        self.parser.add_argument(
            "--icl_eval",
            action="store_true",
            help="If the evaluation should be performed based on the ICL judge prompt.",
        )
        self.parser.add_argument(
            "--icl_model",
            type=str,
            default="gpt-5",
            help="Model used only for ICL judging. This is separate from --model, "
            "which is the model being evaluated for contamination.",
        )
        self.parser.add_argument(
            "--icl_base_url",
            type=str,
            default=None,
            help="Optional OpenAI-compatible base URL for the ICL judge. "
            "If omitted, the official OpenAI API is used with OPENAI_API_KEY.",
        )
        self.parser.add_argument(
            "--icl_max_tokens",
            type=int,
            default=1000,
            help="Maximum tokens for each ICL judge response.",
        )
        self.parser.add_argument(
            "--process_guided_replication",
            required=True,
            action="store_true",
            help="Whether to perform replication using guided instructions. "
            "If false, guided replication is disabled. When provided without "
            "--process_general_replication, it only performs ICL "
            "evaluation.",
        )
        self.parser.add_argument(
            "--process_general_replication",
            action="store_true",
            help="Whether to perform replication using general instructions. "
            "If false, general replication is disabled, and therefore, "
            "evaluations based on BLEURT and ROUGE-L cannot be performed unless"
            "'generated_general_completions' and 'generated_guided_completions'"
            "columns are already provided in the csv file.",
        )
        self.parser.add_argument(
            "--experiment",
            type=str,
            required=True,
            help="The name of the experiment. All final results will be saved in this directory.",
        )

    def check_text_column(self, args):
        if args.task == "nli" and len(args.text_column) < 2:
            self.parser.error(
                "For an NLI-based dataset, two columns should be provided, "
                "corresponding to sentence1/premise and sentence2/hypothesis, "
                "respectively, separated by a space."
            )

        if len(args.text_column) > 2:
            self.parser.error(
                "Exceeded maximum allowed arguments for text_column. "
                "You should provide 1 string for single-instance datasets, "
                "or 2 strings for double-instance datasets, "
                "separated by a space."
            )

    def check_label_column(self, args):
        if args.task in ["nli", "cls"] and not args.label_column:
            self.parser.error(
                "The '--label_column' argument is required when the task "
                "is 'nli' or 'cls'."
            )

    def check_text_split_params(self, args):
        if not 0 <= args.min_p <= 100:
            raise self.parser.error("Minimum percentage should be between 0 and 100.")

        if not 0 <= args.max_p <= 100:
            raise self.parser.error("Maximum percentage should be between 0 and 100.")

        if args.min_p > args.max_p:
            raise self.parser.error(
                "Minimum percentage should be smaller or equal to "
                "maximum percentage."
            )

    def parse_args(self):
        args = self.parser.parse_args()
        self.check_text_column(args)
        self.check_label_column(args)
        self.check_text_split_params(args)
        return args
