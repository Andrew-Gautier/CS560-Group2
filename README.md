# CS560-Group2
  
## Getting Started: Clone the Repository

1. Create a folder for your project (optional):

	**macOS & Windows:**
	```bash
	mkdir CS560-Group2
	cd CS560-Group2
	```

2. Clone the repository:

	```bash
	git clone <repository-url>
	cd CS560-Group2
	```

Replace `<repository-url>` with the actual URL of this repository (e.g., `https://github.com/Andrew-Gautier/CS560-Group2.git`).

## Python Environment Setup (macOS & Windows)

### 1. Create a Virtual Environment

#### macOS & Windows (using Python 3)

Open a terminal (macOS) or Command Prompt (Windows) in the project directory and run:

```bash
python3 -m venv .venv
```

On Windows, you may use `python` instead of `python3` if needed:

```cmd
python -m venv .venv
```

### 2. Activate the Virtual Environment

#### macOS
```bash
source .venv/bin/activate
```

#### Windows (Command Prompt)
```cmd
.venv\Scripts\activate
```

#### Windows (PowerShell)
```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

With the virtual environment activated, run:

```bash
pip install -r requirements.txt
```

### 4. Deactivate the Environment

When finished, deactivate with:

```bash
deactivate
```
