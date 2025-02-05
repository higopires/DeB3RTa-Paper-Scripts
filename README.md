# DeB3RTa: A Transformer-based Model for the Portuguese Financial Domain

## Authors
Higo Pires, Leonardo Paucar, Joao Paulo Carvalho

## Abstract
The complex and specialized terminology of financial language in Portuguese-speaking markets creates significant challenges for natural language processing (NLP) applications, which must capture nuanced linguistic and contextual information to support accurate analysis and decision-making. This paper presents DeB3RTa, a transformer-based model specifically developed through a mixed-domain pretraining strategy that combines extensive corpora from finance, politics, business management, and accounting to enable a nuanced understanding of financial language. 

DeB3RTa was evaluated against prominent models—including BERTimbau, XLM-RoBERTa, SEC-BERT, BusinessBERT, and GPT-based variants—and consistently achieved significant gains across key financial NLP benchmarks. To maximize adaptability and accuracy, DeB3RTa integrates advanced fine-tuning techniques such as layer reinitialization, mixout regularization, stochastic weight averaging, and layer-wise learning rate decay, which together enhance its performance across varied and high-stakes NLP tasks. 

These findings underscore the efficacy of mixed-domain pretraining in building high-performance language models for specialized applications. With its robust performance in complex analytical and classification tasks, DeB3RTa offers a powerful tool for advancing NLP in the financial sector and supporting nuanced language processing needs in Portuguese-speaking contexts.

## Repository Structure
The repository is organized as follows:

- **`Scripts/`**: Contains all scripts necessary to execute the code used in the paper.
- **`Data/`**: Includes the datasets used for model training and evaluation.

To ensure proper execution, both the script file and dataset CSV files must be located in the same directory.

## Usage Instructions

### Running Transformer-based Models
To train or evaluate transformer-based models, execute the script with the following command:
```bash
python <script_name>.py --m <model_name_or_path>
```

Where:
- `<script_name>.py` is the name of the script you want to run.
- `<model_name_or_path>` is either the name of the model available in the Hugging Face Model Hub (e.g., `FacebookAI/xlm-roberta-base`) or the path to a locally stored model.

Example:
```bash
python seq_classification_16_32.py --m FacebookAI/xlm-roberta-base
```

### Running GPT-based Models
For GPT-based models, an **OpenAI API key** is required. Before executing the script, store your API key as an environment variable:
```bash
export OPENAI_API_KEY="your_api_key_here"
```
Then, execute the script:
```bash
python <script_name>.py
```
Ensure that the API key is correctly set, as the script will use it to interact with OpenAI's API.

## Citation
If you use DeB3RTa in your research, please cite our paper:
```
@article{pires2024deb3rta,
  title={DeB3RTa: A Transformer-based Model for the Portuguese Financial Domain},
  author={Pires, Higo and Paucar, Leonardo and Carvalho, Joao Paulo},
  journal={Big Data and Cognitive Computing},
  year={2025}
}
```

## License
This repository is released under the [MIT License](LICENSE).

---
For questions or issues, please contact [higo.pires@ifma.edu.br].
