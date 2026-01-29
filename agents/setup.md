# Ollama Setup

## Installing Ollama in a customized directory

Replace `<user>` by the actual user directory:

~~~bash
# 1. Create your custom directory
mkdir -p /home/<user>/bin/ollama

# 2. Get the latest release URL (replace v0.9.6 with the latest version)
cd /home/<user>/bin/ollama

# 3. Download the Linux amd64 archive
curl -L https://github.com/ollama/ollama/releases/download/v0.9.6/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz

# 4. Extract the archive
tar -xzf ollama-linux-amd64.tgz

# 5. Make the binary executable (if needed)
chmod +x bin/ollama

# 6. Clean up the archive
rm ollama-linux-amd64.tgz

# 7. Add to PATH
echo 'export PATH="/home/<user>/bin/ollama:$PATH"' >> ~/.bashrc
echo 'export OLLAMA_MODELS="/home/<user>/bin/ollama/models"' >> ~/.bashrc
source ~/.bashrc
~~~

Running the server:

~~~bash
./ollama serve
~~~

## Creating a Virtual Environment for Ollama

Inside the root folder `/ollama`:
~~~bash
python3 -m venv .venv
~~~

## Running a Virtual Environment for Ollama

Inside the root folder `/ollama`:
~~~bash
source .venv/bin/activate
~~~

~~~bash
pip install -r requirements.txt
~~~

