import time

import pandas as pd
from tqdm import tqdm

from helpers.experiment_result_saver import ExperimentResultSaver
from helpers.logging_config import configure_logger
from helpers.text_helper import split_text_randomly
from services.openai_api import OpenAIClient

logger = configure_logger(__name__)


class ReplicationPhase(ExperimentResultSaver):
    def __init__(self, df, args, instruction, save_intermediate_results):
        super().__init__(df, args.filepath, args.experiment, save_intermediate_results)
        self.df = df
        self.args = args
        self.instruction = instruction
        self.instruction_type = str(instruction.__class__.__name__).lower()
        self.generated_text_column = f"generated_{self.instruction_type}_completion"
        self.openai_client = OpenAIClient(base_url=getattr(args, 'base_url', None))
        self.sleep_time = getattr(args, 'sleep_time', 3.0)

    def split_text(self):
        if self.args.task == "nli" or all(
            item in self.df.columns for item in ["first_piece", "second_piece"]
        ):
            return
        elif (
            self.args.task != "nli"
            and not all(
                item in self.df.columns for item in ["first_piece", "second_piece"]
            )
            and not self.args.should_split_text
        ):
            raise ValueError(
                "For generating completions for single-instance datasets, "
                "the text must be splitted randomly. If you have pre-split "
                "text, ensure they are listed as 'first_piece' and "
                "'second_piece' columns in the csv file. Otherwise, you can "
                "get the text splitted by running --should_split_text."
            )

        self.df[["first_piece", "second_piece"]] = (
            self.df[self.args.text_column[0]]
            .apply(
                split_text_randomly,
                min_p=self.args.min_p,
                max_p=self.args.max_p,
                seed=self.args.seed,
            )
            .apply(pd.Series)
        )

    def process(self):
        logger.info(f"Starting {self.instruction_type} replication process ...")

        self.split_text()

        with tqdm(total=len(self.df), desc="Generating completions") as pbar:
            for index, row in self.df.iterrows():
                self._perform_task(index, row)
                pbar.update(1)
                time.sleep(self.sleep_time)

            pbar.close()
            self.save_to_csv()

        return self.df

    def _perform_task(self, index, row):
        prompt = self.instruction.get_prompt(self.args.task)
        first_piece = (
            row[self.args.text_column[0]]
            if self.args.task == "nli"
            else row["first_piece"]
        )

        formatted_prompt = self._prepare_prompt(prompt, row, first_piece)

        if index == 0:
            logger.info(f"Input prompt:\n\n{formatted_prompt}")

        thinking_mode = getattr(self.args, 'thinking_mode', False)
        thinking_budget = getattr(self.args, 'thinking_budget', 1000)
        # Resolve effective max_tokens: use explicit --max_tokens if provided,
        # otherwise default to thinking_budget+1000 (thinking) or 500 (normal).
        explicit_max_tokens = getattr(self.args, 'max_tokens', None)
        if explicit_max_tokens is not None:
            max_tokens = explicit_max_tokens
        elif thinking_mode:
            max_tokens = 20000
        else:
            max_tokens = 500

        # In thinking mode, use a brevity system message unless the user
        # provided their own via --system_message.  This prevents the model
        # from spiralling into dozens of pages of reasoning before answering.
        user_system_message = getattr(self.args, 'system_message', None)
        if thinking_mode and not user_system_message:
            system_message = (
                "You are a concise text-completion assistant. "
                "Reason in at most 2–3 short sentences, then output ONLY "
                "the exact completion text that was requested. "
                "Do not explain, analyse, or add any commentary."
            )
        else:
            system_message = user_system_message

        self.df.at[index, self.generated_text_column] = self.openai_client.get_text(
            text=formatted_prompt,
            model=self.args.model,
            max_tokens=max_tokens,
            system_message=system_message,
            thinking_mode=thinking_mode,
            thinking_budget=thinking_budget,
        )

    def _prepare_prompt(self, prompt, row, first_piece):
        if self.args.label_column:
            formatted_prompt = prompt.format(
                split_name=self.args.split,
                dataset_name=self.args.dataset,
                label=str(row[self.args.label_column]),
                first_piece=str(first_piece),
            )
        else:
            formatted_prompt = prompt.format(
                split_name=self.args.split,
                dataset_name=self.args.dataset,
                first_piece=str(first_piece),
            )
        return formatted_prompt
