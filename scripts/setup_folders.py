import os

def create_structure():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Define directories to create
    directories = [
        # Data subdirectories
        "data/raw/sentinel",
        "data/raw/dynamic_world",
        "data/raw/osm",
        "data/interim",
        "data/processed",
        "data/tiles",
        "data/features",
        
        # Artifacts subdirectories
        "artifacts/models",
        "artifacts/scalers",
        "artifacts/shap",
        "artifacts/metrics",
        
        # Logs
        "logs",
        
        # Deployment subdirectories
        "deployment/docker",
        "deployment/oci",
        "deployment/ci_cd",
        
        # Documentation subdirectories
        "docs/architecture",
        "docs/api",
        "docs/diagrams",
        "docs/reports",
        "docs/references",
        
        # Image assets
        "images/architecture",
        "images/dashboard",
        "images/outputs",
        "images/readme",
        
        # Source modules
        "gee",
        "osm",
        "feature_engineering",
        "deep_learning",
        "ml_models",
        "explainability",
        "dashboard",
        "api",
        "config",
        "scripts",
        "tests",
        "notebooks"
    ]
    
    # Python packages that need __init__.py
    packages = [
        "gee",
        "osm",
        "feature_engineering",
        "deep_learning",
        "ml_models",
        "explainability",
        "dashboard",
        "api",
        "config",
        "scripts",
        "tests"
    ]
    
    print("Initializing project directory structure...")
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {directory}")
        
        # If it is a data, artifact, log, or doc directory, add a .gitkeep so git tracks it
        if any(directory.startswith(prefix) for prefix in ["data", "artifacts", "logs", "docs", "images", "notebooks"]):
            gitkeep_path = os.path.join(dir_path, ".gitkeep")
            if not os.path.exists(gitkeep_path):
                with open(gitkeep_path, "w") as f:
                    pass
                print(f"  Added .gitkeep to {directory}")

    print("\nInitializing python package files (__init__.py)...")
    for package in packages:
        init_path = os.path.join(base_dir, package, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                f.write(f'"""\n{package.replace("_", " ").title()} package module.\n"""\n')
            print(f"Created package initializer: {package}/__init__.py")

    print("\nProject structure initialized successfully.")

if __name__ == "__main__":
    create_structure()
