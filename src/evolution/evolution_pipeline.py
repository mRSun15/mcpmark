"""
Evolution Pipeline
==================

Main orchestrator for self-evolving agent experiments.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.logger import get_logger
from src.factory import MCPServiceFactory
from src.model_config import ModelConfig
from src.results_reporter import ResultsReporter, TaskResult
from src.agents import AGENT_REGISTRY

from .task_split import EVOLUTION_TASKS
from .feedback_agent import FeedbackAgent, FailureAnalyzer
from .prompt_evolver import PromptEvolutionAgent
from .prompt_engineer import PromptEngineerAgent

logger = get_logger(__name__)


class EvolutionPipeline:
    """Main orchestrator for self-evolving agent experiments."""
    
    def __init__(
        self,
        mcp_service: str,
        model: str,
        timeout: int = 3600,
        exp_name: str = "evolution-exp",
        output_dir: Path = Path("./results"),
        reasoning_effort: str = "default",
        agent_name: str = "mcpmark",
    ):
        self.mcp_service = mcp_service
        self.model_name = model
        self.timeout = timeout
        self.exp_name = exp_name
        self.reasoning_effort = reasoning_effort
        self.agent_name = agent_name
        
        model_config = ModelConfig(self.model_name)
        self.api_key = model_config.api_key
        self.base_url = model_config.base_url
        self.litellm_input_model_name = model_config.litellm_input_model_name
        self.litellm_run_model_name = None
        
        self.task_manager = MCPServiceFactory.create_task_manager(mcp_service)
        self.state_manager = MCPServiceFactory.create_state_manager(mcp_service)
        self.service_config = self.state_manager.get_service_config_for_agent()
        self.results_reporter = ResultsReporter()
        
        model_slug = self.model_name.replace(".", "-")
        if self.reasoning_effort != "default":
            model_slug += f"-{self.reasoning_effort}"
        
        self.base_dir = output_dir / exp_name / f"{model_slug}__{mcp_service}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.feedback_agent = FeedbackAgent(
            model=self.litellm_input_model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            reasoning_effort=self.reasoning_effort,
            timeout=timeout // 4,
        )
        
        self.prompt_evolver = PromptEvolutionAgent(
            model=self.litellm_input_model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            reasoning_effort=self.reasoning_effort,
            timeout=timeout // 4,
        )
        
        self.prompt_engineer = PromptEngineerAgent(
            model=self.litellm_input_model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            reasoning_effort=self.reasoning_effort,
            timeout=timeout // 2,  # Increased for high-latency connections
        )
        
        self.failure_analyzer = FailureAnalyzer(
            model=self.litellm_input_model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            reasoning_effort=self.reasoning_effort,
            timeout=timeout // 4,
        )
        
        self.agent = None
        self._create_agent()

    def _create_agent(
        self,
        evolved_prompts: Optional[Dict[str, Dict]] = None,
        enable_prompt_engineer: bool = False,
        feedback: Optional[Dict[str, Any]] = None,
    ):
        agent_cls = AGENT_REGISTRY[self.agent_name]
        self.agent = agent_cls(
            litellm_input_model_name=self.litellm_input_model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            mcp_service=self.mcp_service,
            timeout=self.timeout,
            service_config=self.service_config,
            service_config_provider=self.state_manager.get_service_config_for_agent,
            reasoning_effort=self.reasoning_effort,
        )
        
        if evolved_prompts:
            self._inject_evolved_prompts(evolved_prompts)
        
        if enable_prompt_engineer:
            if feedback:
                self.prompt_engineer.set_feedback(feedback)
            else:
                self.prompt_engineer.clear_feedback()
            self.agent.set_prompt_engineer(self.prompt_engineer, enabled=True)

    def _inject_evolved_prompts(self, evolved_prompts: Dict[str, Dict]):
        prompt_texts = {role: ep["evolved_prompt"] for role, ep in evolved_prompts.items()}
        self.agent.set_evolved_prompts(prompt_texts)

    def _get_base_prompts(self) -> Dict[str, str]:
        return self.agent.get_default_prompts()

    def _format_duration(self, seconds: float) -> str:
        return f"{(seconds * 1000):.2f}ms" if seconds < 1 else f"{seconds:.2f}s"

    def _get_task_output_dir(self, task_name: str, set_label: str) -> Path:
        return self.base_dir / set_label / task_name

    def _load_task_result(self, task_name: str, set_label: str) -> Optional[Dict[str, Any]]:
        meta_path = self._get_task_output_dir(task_name, set_label) / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _run_single_task(self, task, set_label: str) -> TaskResult:
        task_start_time = time.time()
        
        logger.info(f"\n┌─ Stage 1: Setup")
        setup_success = self.state_manager.set_up(task)
        if not setup_success:
            return TaskResult(
                task_name=task.name, success=False, error_message="State Duplication Error",
                verification_error=None, verification_output=None,
                category_id=task.category_id, task_id=task.task_id,
                agent_execution_time=0.0, task_execution_time=time.time() - task_start_time,
            )
        
        logger.info(f"┌─ Stage 2: Execute")
        exec_start = time.time()
        task_instruction = self.task_manager.get_task_instruction(task)
        task_output_dir = self._get_task_output_dir(task.name, set_label)
        task_output_dir.mkdir(parents=True, exist_ok=True)
        execution_log_path = task_output_dir / "execution.log"
        if execution_log_path.exists():
            execution_log_path.unlink()
        
        agent_result = self.agent.execute_sync(task_instruction, str(execution_log_path))
        agent_execution_time = time.time() - exec_start
        
        if agent_result.get("litellm_run_model_name"):
            self.litellm_run_model_name = agent_result["litellm_run_model_name"]
        
        messages_path = task_output_dir / "messages.json"
        self.results_reporter.save_messages_json(agent_result.get("output", []), messages_path)
        self.state_manager.set_verification_environment(str(messages_path))
        
        logger.info(f"┌─ Stage 3: Verify")
        try:
            result = self.task_manager.execute_task(task, agent_result)
        finally:
            import os
            os.environ.pop("MCP_MESSAGES", None)
            os.environ.pop("MCP_GITHUB_TOKEN", None)
        
        logger.info(f"┌─ Stage 4: Cleanup")
        self.state_manager.clean_up(task)
        
        result.agent_execution_time = agent_execution_time
        result.task_execution_time = time.time() - task_start_time
        
        meta_path = task_output_dir / "meta.json"
        model_config = {
            "mcp_service": self.mcp_service, "model_name": self.model_name,
            "litellm_run_model_name": self.litellm_run_model_name,
            "reasoning_effort": self.reasoning_effort, "timeout": self.timeout,
            "agent_name": self.agent_name, "evolution_set": set_label,
        }
        self.results_reporter.save_meta_json(
            result, model_config, datetime.fromtimestamp(task_start_time), datetime.now(), meta_path,
        )
        return result

    def _read_execution_log(self, task_name: str, set_label: str) -> str:
        log_path = self._get_task_output_dir(task_name, set_label) / "execution.log"
        return log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    def _save_error_meta(self, task_name: str, set_label: str, error_message: str):
        task_output_dir = self._get_task_output_dir(task_name, set_label)
        task_output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = task_output_dir / "meta.json"
        error_meta = {
            "task_name": task_name, "success": False, "error_message": error_message,
            "crashed": True, "mcp_service": self.mcp_service, "model_name": self.model_name,
            "agent_name": self.agent_name, "evolution_set": set_label,
            "timestamp": datetime.now().isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(error_meta, f, ensure_ascii=False, indent=2)

    def _get_task_by_name(self, task_name: str):
        category_id, task_id = task_name.split("__", 1)
        tasks = self.task_manager.filter_tasks(f"{category_id}/{task_id}")
        if tasks:
            return tasks[0]
        raise ValueError(f"Task not found: {task_name}")

    def run_evolution_experiment(self) -> Dict[str, Any]:
        """
        Run evolution experiment:
        1. Run all tasks once with PromptEngineer
        2. Generate feedback for failed tasks only
        """
        experiment_start = time.time()
        logger.info("\n" + "=" * 80)
        logger.info("EVOLUTION EXPERIMENT STARTED")
        logger.info("=" * 80)
        
        base_prompts = self._get_base_prompts()
        with open(self.base_dir / "base_prompts.json", "w") as f:
            json.dump(base_prompts, f, indent=2)
        
        all_tasks = EVOLUTION_TASKS
        logger.info(f"Running {len(all_tasks)} tasks")
        
        # Run all tasks once
        results: List[Dict[str, Any]] = []
        task_feedbacks: List[Dict[str, Any]] = []
        
        for i, task_name in enumerate(all_tasks, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Task {i}/{len(all_tasks)}: {task_name}")
            logger.info("=" * 60)
            
            # Check for existing result (resume support)
            existing = self._load_task_result(task_name, "run")
            task_data = None
            
            if existing and not existing.get("crashed", False):
                logger.info(f"[Run] Resumed - {'PASSED' if existing.get('success') else 'FAILED'}")
                results.append({"task_name": task_name, "passed": existing.get("success", False)})
                task_data = {
                    "task_name": task_name,
                    "passed": existing.get("success", False),
                    "error_message": existing.get("error_message"),
                    "verification_output": existing.get("verification_output", ""),
                    "execution_log": self._read_execution_log(task_name, "run"),
                }
            else:
                logger.info(f"[Run] Executing")
                self._create_agent(enable_prompt_engineer=True, feedback=None)
                self.prompt_engineer.clear_generated_prompts()
                try:
                    task = self._get_task_by_name(task_name)
                    result = self._run_single_task(task, "run")
                    results.append({"task_name": task_name, "passed": result.success})
                    task_data = {
                        "task_name": task_name,
                        "passed": result.success,
                        "error_message": result.error_message,
                        "verification_output": result.verification_output,
                        "execution_log": self._read_execution_log(task_name, "run"),
                    }
                    # Save generated prompts
                    prompts = self.prompt_engineer.get_generated_prompts()
                    if prompts:
                        p = self.base_dir / "run" / task_name / "generated_prompts.json"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        with open(p, "w") as f:
                            json.dump(prompts, f, indent=2)
                    logger.info(f"[Run] {'PASSED' if result.success else 'FAILED'}")
                except Exception as e:
                    logger.error(f"[Run] Error: {e}")
                    results.append({"task_name": task_name, "passed": False})
                    task_data = {"task_name": task_name, "passed": False, "error_message": str(e)}
                    self._save_error_meta(task_name, "run", str(e))
            
            # Generate feedback for failed tasks only
            if not task_data["passed"]:
                fb_path = self.base_dir / "run" / task_name / "task_feedback.json"
                if fb_path.exists():
                    try:
                        with open(fb_path) as f:
                            feedback = json.load(f)
                        task_feedbacks.append({"task_name": task_name, **feedback})
                        logger.info("[Feedback] Loaded existing feedback")
                    except Exception:
                        pass
                else:
                    try:
                        logger.info("[Feedback] Generating feedback for failed task")
                        fb = self.feedback_agent.task_feedback_agent.generate_task_feedback(
                            task_data, base_prompts
                        )
                        fb_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(fb_path, "w") as f:
                            json.dump(fb, f, indent=2)
                        task_feedbacks.append({"task_name": task_name, **fb})
                        logger.info("[Feedback] Generated and saved")
                    except Exception as e:
                        logger.error(f"[Feedback] Error: {e}")
                        task_feedbacks.append({"task_name": task_name, "error": str(e)})
        
        # Calculate results
        n = len(all_tasks)
        success_count = sum(1 for r in results if r["passed"])
        success_rate = success_count / n if n > 0 else 0
        
        # Analyze failure patterns
        failure_analysis = {}
        if task_feedbacks:
            logger.info("\n" + "=" * 60)
            logger.info("ANALYZING FAILURE PATTERNS")
            logger.info("=" * 60)
            try:
                failure_analysis = self.failure_analyzer.analyze_failures(task_feedbacks)
                # Save failure analysis separately
                with open(self.base_dir / "failure_analysis.json", "w") as f:
                    json.dump(failure_analysis, f, indent=2)
                logger.info(f"[Analysis] Found {len(failure_analysis.get('clusters', []))} failure clusters")
                logger.info(f"[Analysis] Summary: {failure_analysis.get('summary', 'N/A')}")
            except Exception as e:
                logger.error(f"[Analysis] Error: {e}")
                failure_analysis = {"error": str(e)}
        
        experiment_result = {
            "experiment_name": self.exp_name,
            "model": self.model_name,
            "mcp_service": self.mcp_service,
            "tasks": all_tasks,
            "success_count": success_count,
            "success_rate": success_rate,
            "results": results,
            "failed_task_feedbacks": task_feedbacks,
            "failure_analysis": failure_analysis,
            "experiment_time": time.time() - experiment_start,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(self.base_dir / "evolution_result.json", "w") as f:
            json.dump(experiment_result, f, indent=2)
        
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT COMPLETED")
        logger.info(f"Success: {success_count}/{n} ({success_rate:.1%})")
        logger.info(f"Feedback generated for {len(task_feedbacks)} failed tasks")
        if failure_analysis.get("clusters"):
            logger.info(f"Failure clusters: {len(failure_analysis['clusters'])}")
            for cluster in failure_analysis["clusters"]:
                logger.info(f"  - {cluster['cluster_name']}: {len(cluster['affected_tasks'])} tasks")
        logger.info(f"Time: {self._format_duration(time.time() - experiment_start)}")
        logger.info("=" * 80)
        
        return experiment_result
