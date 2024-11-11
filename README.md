# Vietnamese Administrative Units AI Training Program

This repository contains a program to train an AI model to understand and answer questions about the Vietnamese administrative units. The project includes scripts for data extraction from a SQL Server database, data preprocessing, model training using Hugging Face's Transformers library, and pushing the trained model to the Hugging Face Hub.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Database Setup](#database-setup)
  - [Using Python (`extract.py`)](#using-python-extractpy)
  - [Using Node.js and TypeORM (`extract.ts`)](#using-nodejs-and-typeorm-extractts)
- [Training the Model (`main.py`)](#training-the-model-mainpy)
- [Pushing the Model to Hugging Face](#pushing-the-model-to-hugging-face)
- [Testing the Model (`test_model.py`)](#testing-the-model-test_modelpy)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- Extracts data from a SQL Server database containing Vietnamese administrative units.
- Generates question-answer pairs based on the administrative hierarchy.
- Fine-tunes a language model (GPT-2 Medium) to understand and answer questions about administrative units.
- Pushes the trained model to the Hugging Face Hub for easy sharing and deployment.
- Provides test scripts to evaluate the trained model.

## Prerequisites

- **Python 3.7+**
- **Node.js 14+** (if using the Node.js extract script)
- **SQL Server Database** containing the `Province`, `District`, and `Ward` tables.
- **GPU** with sufficient VRAM (if training on GPU)
- **Hugging Face Account** and API Token

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your_username/vietnamese-admin-address.git
   cd vietnamese-admin-address
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install Python Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   **Note**: If `requirements.txt` is not provided, install the necessary packages manually:

   ```bash
   pip install transformers datasets huggingface_hub sqlalchemy pyodbc pandas torch
   ```

4. **Install Node.js Dependencies** (if using the Node.js extract script)

   ```bash
   cd node_extractor  # Navigate to the directory containing the Node.js script
   npm install
   ```

## Database Setup

Ensure you have a SQL Server database with the following tables: `Province`, `District`, and `Ward`. The database schema should match the design specified in the `SQL Design` section.

### Using Python (`extract.py`)

1. **Install ODBC Driver for SQL Server**

   **For Ubuntu 22.04:**

   ```bash
   curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
   curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list | sudo tee /etc/apt/sources.list.d/mssql-release.list
   sudo apt update
   sudo ACCEPT_EULA=Y apt install -y msodbcsql18 unixodbc-dev
   ```

2. **Configure Database Connection**

   Update the database connection details in `extract.py`:

   ```python
   # extract.py

   # Database connection details
   server = "your_server_name"
   database = "your_database_name"
   username = "your_username"
   password = "your_password"
   ```

3. **Run the Extract Script**

   ```bash
   python extract.py
   ```

   This will generate `vietnamese_administrative_units_train.jsonl` containing the question-answer pairs for training.

### Using Node.js and TypeORM (`extract.ts`)

1. **Install Dependencies**

   ```bash
   npm install typeorm reflect-metadata mssql
   npm install typescript ts-node @types/node --save-dev
   ```

2. **Configure Database Connection**

   Update the connection details in `src/data-source.ts`:

   ```typescript
   // src/data-source.ts

   const server = 'your_server_name';
   const database = 'your_database_name';
   const username = 'your_username';
   const password = 'your_password';
   ```

3. **Run the Extract Script**

   ```bash
   npx ts-node src/extract.ts
   ```

   This will generate `vietnamese_administrative_units_train.json` with the training data.

## Training the Model (`main.py`)

1. **Set Up Hugging Face Authentication**

   Obtain your Hugging Face API token from [Hugging Face Tokens](https://huggingface.co/settings/tokens) and set it as an environment variable:

   ```bash
   export HUGGINGFACE_TOKEN=your_hf_token
   ```

2. **Configure Training Script**

   Update the Hugging Face username in `main.py`:

   ```python
   # main.py

   hf_username = "your_hf_username"  # Replace with your Hugging Face username
   ```

3. **Run the Training Script**

   ```bash
   python main.py
   ```

   This script will:

   - Load the training data from `vietnamese_administrative_units_train.jsonl` or `.json`.
   - Fine-tune the GPT-2 Medium model on the data.
   - Push the trained model to the Hugging Face Hub.

## Pushing the Model to Hugging Face

The `main.py` script automatically pushes the trained model to the Hugging Face Hub if `push_to_hub=True` is set in `TrainingArguments`.

Ensure that:

- You are logged in to Hugging Face using `huggingface_hub.login()`.
- The `hub_model_id` is correctly set to your desired repository name.

Example in `main.py`:

```python
training_args = TrainingArguments(
    output_dir=model_path,
    eval_strategy="epoch",
    push_to_hub=True,
    hub_model_id=f"{hf_username}/vietnamese-administrative-units-model",
    hub_token=hf_token,
    # ... other arguments ...
)
```

## Testing the Model (`test_model.py`)

1. **Create the Test Script**

   Ensure `test_model.py` contains the following code:

   ```python
   import torch
   from transformers import AutoTokenizer, AutoModelForCausalLM

   model_path = "./admin_unit_model_gpt2_medium"  # Or your Hugging Face model ID
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   model = AutoModelForCausalLM.from_pretrained(model_path)

   device = "cuda" if torch.cuda.is_available() else "cpu"
   model.to(device)

   def generate_response(prompt, max_length=128):
       inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
       output = model.generate(
           inputs["input_ids"],
           attention_mask=inputs["attention_mask"],
           max_length=max_length,
           num_beams=4,
           early_stopping=True,
           pad_token_id=tokenizer.eos_token_id,
       )
       response = tokenizer.decode(output[0], skip_special_tokens=True)
       return response

   # Example usage
   prompt = "How many provinces are there in Vietnam?"
   print(f"Prompt: {prompt}")
   print(f"Response: {generate_response(prompt)}")
   ```

2. **Run the Test Script**

   ```bash
   python test_model.py
   ```

## Usage Examples

You can use the trained model to answer various questions about Vietnamese administrative units.

**Example 1:**

```python
prompt = "How many districts are in Hanoi?"
response = generate_response(prompt)
print(response)
```

**Example 2:**

```python
prompt = "What administrative level is 'Dong Da'?"
response = generate_response(prompt)
print(response)
```

## Troubleshooting

- **PyTorch Not Found**: If you encounter `ModuleNotFoundError: No module named 'torch'`, install PyTorch:

  ```bash
  pip install torch
  ```

- **CUDA Compatibility Issues**: If you face CUDA errors, ensure that your PyTorch installation matches your CUDA version or use the CPU-only version.

- **Database Connection Errors**: Verify your database connection details and ensure that the SQL Server allows remote connections.

- **Deprecated Warnings**: Update deprecated arguments as per the latest Transformers documentation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
