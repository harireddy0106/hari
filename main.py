# main.py

import os
import zipfile

# Define your folder structure
structure = {
    "floatchat-argo-ai": {
        "README.md": "",
        "requirements.txt": "",
        "Dockerfile": "",
        "docker-compose.yml": "",
        ".env.example": "",
        ".gitignore": "",
        "setup.py": "",
        "src": {
            "__init__.py": "",
            "config.py": "",
            "main.py": "",
            "database": {
                "__init__.py": "",
                "database_manager.py": "",
                "models.py": "",
                "migrations": {
                    "__init__.py": "",
                    "001_initial.py": ""
                }
            },
            "data": {
                "__init__.py": "",
                "argo_data_ingestor.py": ""
            },
            "nlp": {
                "__init__.py": "",
                "query_processor.py": ""
            },
            "visualization": {
                "__init__.py": "",
                "plot_generator.py": ""
            },
            "utils": {
                "__init__.py": "",
                "helpers.py": ""
            }
        },
        "streamlit_app": {
            "__init__.py": "",
            "main.py": ""
        },
        "scripts": {
            "setup.sh": "",
            "initialize_database.py": "",
            "download_sample_data.py": "",
            "deploy.py": ""
        },
        "config": {
            "__init__.py": "",
            "settings.yaml": ""
        },
        "tests": {
            "__init__.py": "",
            "test_database.py": "",
            "test_ingestion.py": "",
            "test_nlp.py": ""
        },
        "docs": {
            "installation.md": "",
            "usage.md": "",
            "api.md": ""
        }
    }
}

def create_structure(base_path, struct):
    for name, content in struct.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, 'w') as f:
                f.write(content)

# Create folder structure
base_folder = "floatchat-argo-ai"
create_structure(".", structure[base_folder])

# Create zip file
zip_filename = f"{base_folder}.zip"
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            zipf.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file), base_folder))

print(f"Zip file '{zip_filename}' created successfully!")
