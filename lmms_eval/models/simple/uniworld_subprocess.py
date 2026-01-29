"""
UniWorld model wrapper using subprocess for environment isolation.
Avoids dependency conflicts by running UniWorld in its own environment.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("uniworld_subprocess")
class UniWorldSubprocess(lmms):
    """
    UniWorld via subprocess - runs in isolated environment
    
    Setup:
    1. Create separate conda environment for UniWorld:
       conda create -n uniworld python=3.10
       conda activate uniworld
       cd UniWorld && pip install -r requirements.txt
    
    2. Use this wrapper in lmms-eval:
       python -m lmms_eval \
           --model uniworld_subprocess \
           --model_args pretrained=./UniWorld-V1,conda_env=uniworld,script_path=./UniWorld/app.py
    """

    def __init__(
        self,
        pretrained: str,
        conda_env: str = "uniworld",
        script_path: Optional[str] = None,
        output_image_dir: Optional[str] = "./uniworld_subprocess_images",
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.pretrained = pretrained
        self.conda_env = conda_env
        self.batch_size = batch_size
        
        # Find UniWorld script
        if script_path is None:
            # Auto-detect
            wd = Path(__file__).parent.parent.parent.parent.resolve()
            possible_paths = [
                wd / "UniWorld" / "UniWorld-V1" / "inference.py",
                wd / "UniWorld" / "app.py",
                wd / "UniWorld-V1" / "inference.py",
            ]
            for p in possible_paths:
                if p.exists():
                    self.script_path = str(p)
                    break
            else:
                raise FileNotFoundError(
                    f"UniWorld script not found. Please specify script_path.\n"
                    f"Searched: {[str(p) for p in possible_paths]}"
                )
        else:
            self.script_path = script_path
            
        if not Path(self.script_path).exists():
            raise FileNotFoundError(f"Script not found: {self.script_path}")
            
        eval_logger.info(f"UniWorld script: {self.script_path}")
        eval_logger.info(f"Conda environment: {self.conda_env}")
        
        # Setup output directory
        self.output_image_dir = output_image_dir
        os.makedirs(self.output_image_dir, exist_ok=True)
        
        # Test conda environment
        self._test_environment()
        
    def _test_environment(self):
        """Test if conda environment exists and can run"""
        try:
            result = subprocess.run(
                ["conda", "run", "-n", self.conda_env, "python", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Conda environment '{self.conda_env}' test failed:\n{result.stderr}"
                )
            eval_logger.info(f"✅ Environment test passed: {result.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError(
                "conda command not found. Please ensure conda is installed and in PATH."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Conda environment '{self.conda_env}' test timeout")
    
    def _call_uniworld(
        self,
        prompt: str,
        image_paths: List[str],
        doc_id: str,
        task: str,
        mode: str = "text",
        gen_kwargs: Optional[dict] = None,
    ) -> Tuple[str, List[str]]:
        """
        Call UniWorld via subprocess
        
        Args:
            prompt: Text prompt
            image_paths: List of input image paths
            doc_id: Document ID
            task: Task name
            mode: "text" or "visual_cot"
            gen_kwargs: Generation kwargs (max_new_tokens, temperature, etc.)
            
        Returns:
            (output_text, output_image_paths)
        """
        # Create temporary input file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, encoding='utf-8'
        ) as f:
            input_data = {
                "prompt": prompt,
                "images": image_paths,
                "doc_id": doc_id,
                "task": task,
                "mode": mode,
                "output_dir": self.output_image_dir,
                "gen_kwargs": gen_kwargs or {},
            }
            json.dump(input_data, f, ensure_ascii=False)
            input_file = f.name
            
        # Create temporary output file
        output_file = input_file.replace('.json', '_output.json')
        
        try:
            # Call UniWorld script
            cmd = [
                "conda", "run", "-n", self.conda_env,
                "python", self.script_path,
                "--input", input_file,
                "--output", output_file,
                "--model_path", self.pretrained,
            ]
            
            eval_logger.debug(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
            )
            
            if result.returncode != 0:
                eval_logger.error(f"UniWorld failed:\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}")
                return "", []
            
            # Read output
            if not os.path.exists(output_file):
                eval_logger.error(f"Output file not found: {output_file}")
                return "", []
                
            with open(output_file, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
                
            text = output_data.get("text", "")
            images = output_data.get("images", [])
            
            return text, images
            
        except subprocess.TimeoutExpired:
            eval_logger.error(f"UniWorld timeout for doc_id={doc_id}")
            return "", []
        except Exception as e:
            eval_logger.error(f"UniWorld error: {e}")
            return "", []
        finally:
            # Cleanup temp files
            for f in [input_file, output_file]:
                if os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        """Generate responses for requests"""
        res = []
        pbar = tqdm(total=len(requests), desc="UniWorld (subprocess)")
        
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            prompt = contexts
            
            # Get input images
            doc = self.task_dict[task][split][doc_id]
            image_paths = []
            
            if doc_to_visual is not None:
                visuals = doc_to_visual(doc)
                if not isinstance(visuals, list):
                    visuals = [visuals]
                    
                # Save images to temporary files
                import PIL.Image
                for idx, img in enumerate(visuals):
                    if img is not None and isinstance(img, PIL.Image.Image):
                        temp_path = os.path.join(
                            self.output_image_dir, 
                            f"input_{task}_{doc_id}_{idx}.png"
                        )
                        img.save(temp_path)
                        image_paths.append(temp_path)
            
            # Determine mode
            mode = "visual_cot" if "visual_cot" in task else "text"
            
            # Call UniWorld
            text, images = self._call_uniworld(
                prompt, image_paths, str(doc_id), task, mode, gen_kwargs
            )
            
            # Format output
            output = json.dumps({"text": text, "images": images}, ensure_ascii=False)
            res.append(output)
            
            pbar.update(1)
        
        pbar.close()
        return res
    
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("UniWorld does not support loglikelihood")
    
    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round not implemented")
