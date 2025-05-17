## System-Level Efficient Processing of Instruction-based Llama 3.2-1B
### Project Goal
This project builds an end-to-end model to study both efficient inferencing & finetuning techniques on Meta's Llama 3.2-1B instruct model (more emphasis on finetuning) & evaluate/profile its memory, latency & computational overhead performances on a task-specific dataset. This project does NOT aim to improve the model-level performance, rather it focuses on improving the system-level performance.

This project was part of a graded Master's coursework at USC.

### Inferencing & FineTuning Techniques
- Inferencing: KV Caching
- FineTuning: LoRA, Mixed Precision, Gradient Accumulation, Gradient Checkpointing

### Dataset
We use the open source [Alpaca dataset](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) containing 52K instructions

### GPU Resources
P100, 16GB Memory

### Overview
This project is a minimal, self-contained version of [Meta's LLaMA 3 codebase](https://github.com/meta-llama), redesigned to support both inference and fine-tuning with a focus on simplicity and extensibility. It removes unused dependencies, strips away non-essential features, and introduces modular components tailored for common research workflows.

#### Removed:
- Remove dependencies on the fairscale and fire packages.
- Remove the chat completion feature.
  
#### Customized
Reorganize and simplify the code structure for the generation function. The Generation class is now the base class of the Llama model.

#### Added
- `inference.py`: inference workflow that adds the KV Caching to enable during inferencing & disable during training/finetuning
- `finetuning.py`: finetuning workflow consisting of the main training loop & an argparser CLI support to run with different finetuning techniques, along with the vanilla model
- `project/model/lora.py`: customized LoRA implementation forked from the [official LoRA imlementation](https://github.com/microsoft/LoRA.git)
- `project/model/grad_ckpt.py`: gradient checkpointing implementation
- `project/utils.py`: wandb support for memory, computation & loss profiling
- `alpaca_dataset.py`: customized alpaca dataset for training, forked from the [official Alpaca repository](https://github.com/tatsu-lab/stanford_alpaca)
- `benchmark_inference.py`: to compare the inference performance of the original Meta's Llama3 model (based on system-level) 

### Quick Start
1. Install required packages:
```
pip install -r requirements.txt
```
2. [Request access](https://www.llama.com/llama-downloads/) to Llama models and download the Llama3.2-1B model weights:
    * Install the Llama CLI in your preferred environment: `pip install llama-stack`.
    * Run `llama model list` to show the latest models available.
    * Run `llama model download --source meta --model-id Llama3.2-1B`.
    * When the script prompts for your unique custom URL, copy and paste the custom URL. (Clicking on the URL itself does not access the model)
3. For Finetuning: Run the `finetuning.py` script as below or run `python finetuning.py -h` to refer the help option for more details on how to run different finetuning techniques. Below illustration will train the vanilla model without any finetuning.
```
python finetuning.py --tokenizer_model_path <path-to-your-local-folder>/.llama/checkpoints/Llama3.2-1B/tokenizer.model --checkpoint_path <path-to-your-local-folder>/.llama/checkpoints/Llama3.2-1B/consolidated.00.pth
```
4. For Inferencing: Toggle the `kv_caching` boolean to evaluate the memory & computation performance with & without KV Caching optimization.
   - To check the inference output, run `python inference.py` . 
   - To compare performance with benchmark `Llama3.2-1B/consolidated.00.pth`, run `python benchmark_inference.py` .

Replace the default `consolidated.00.pth` with your saved `finetuned_llama3.2-1B.pth` after running finetuning to compare the finetuned vs non-finetuned outputs. Recommended to enable `kv_caching` boolean during inferencing. 

### Results
Refer `outputs` folder for the output data for both inferencing & finetuning outputs. 
Refer [Metrics Profiling](https://api.wandb.ai/links/asmitamohanty13-usc/g7qh758c) for finetuning results.

1. For Inferencing:
- To evaluate the KV Caching optimization performance. Evaluated on Meta's Llama3 original model `Llama3.2-1B/consolidated.00.pth` without any finetuning or task-specific training.
- Observed **84.4%** reduction in inference time & **29%** reduction in peak memory for a batch size of 16, with 256 input tokens & 32 output tokens
  
2. For Finetuning:
- To evaluate finetuning performance on alpaca dataset with different combinations of finetuning techniques.
- Observed the best performance with LoRA finetuning resulting in **~37%** reduction in both computation time & peak memory wrt vanilla model without any finetuning.


