from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_config import BaseModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_accelerate_available
from lighteval.utils.utils import EnvConfig

if is_accelerate_available():
    from accelerate import Accelerator, InitProcessGroupKwargs

    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))]
    )
else:
    accelerator = None

from ..models.base_model import Model
from .base_benchmark import Benchmark


@dataclass
class LightEvalBenchmark(Benchmark):
    """Base class for LightEval-based benchmarks.

    Attributes:
        tasks: Tasks to evaluate (e.g. "helm|mmlu|5|1")
        output_dir: Directory to save evaluation results
        save_details: Whether to save detailed results
        push_to_hub: Whether to push results to Hugging Face Hub
        hub_results_org: Organization name for Hub results
        cache_dir: Directory for caching
        batch_size: Batch size for evaluation
        max_samples: Maximum number of samples to evaluate (None for all)
        custom_task_directory: Path to custom task definitions
    """

    tasks: str
    output_dir: Path = field(init=False, default=Path("./results"))
    save_details: bool = field(init=False, default=True)
    push_to_hub: bool = field(init=False, default=True)
    hub_results_org: str = field(init=False, default="clembench-project-playpen")
    cache_dir: Path = field(init=False, default=Path("tmp/"))
    batch_size: int = field(init=False, default=8)
    max_samples: int | None = field(init=False, default=100)
    custom_task_directory: Path | None = field(init=False, default=None)

    def _create_evaluation_tracker(self) -> EvaluationTracker:
        """Create the evaluation tracker instance."""
        return EvaluationTracker(
            output_dir=str(self.output_dir),
            save_details=self.save_details,
            push_to_hub=self.push_to_hub,
            hub_results_org=self.hub_results_org,
        )

    def _create_pipeline_params(self) -> PipelineParameters:
        """Create pipeline parameters instance."""
        return PipelineParameters(
            launcher_type=ParallelismManager.ACCELERATE,
            env_config=EnvConfig(cache_dir=str(self.cache_dir)),
            override_batch_size=self.batch_size,
            max_samples=self.max_samples,
            # num_fewshot_seeds=8
        )

    def _create_model_config(self, model: Model) -> BaseModelConfig:
        """Create model configuration for LightEval."""
        return BaseModelConfig(
            pretrained=model.name,
            accelerator=accelerator,  # type: ignore
            dtype="float16",
            use_chat_template=True,
            batch_size=self.batch_size,
            trust_remote_code=False,  # Set this based on your requirements
        )

    def evaluate(self, model: Model) -> dict:
        """Evaluate the model using LightEval.

        Args:
            model: The model to evaluate

        Returns:
            dict: Evaluation results
        """
        evaluation_tracker = self._create_evaluation_tracker()
        pipeline_params = self._create_pipeline_params()
        model_config = self._create_model_config(model)

        pipeline = Pipeline(
            tasks=self.tasks,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model_config=model_config,
            model=model,
        )

        # Run evaluation
        pipeline.evaluate()

        # Save results
        # pipeline.save_and_push_results()

        # Get results
        results = pipeline.show_results()

        return results
