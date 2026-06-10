name: MLflow CI Pipeline

on:
  push:
    branches:
      - main
      - master
  workflow_dispatch:

permissions:
  contents: write

jobs:
  training-job:
    name: Train Model with MLflow
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4
        with:
          lfs: true
          fetch-depth: 0

      - name: Setup Miniconda
        uses: conda-incubator/setup-miniconda@v3
        with:
          auto-update-conda: true
          python-version: "3.10"

      - name: Install MLflow
        shell: bash -l {0}
        run: |
          pip install mlflow==2.16.0
      - name: Run MLProject
        shell: bash -l {0}
        env:
          MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING: "TRUE"
        run: |
          mlflow run ./MLProject --env-manager conda
      - name: Save and Push mlruns to GitHub Repo
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          X_USER_NAME: ${{ secrets.GIT_USER_NAME }}
          X_USER_EMAIL: ${{ secrets.GIT_USER_EMAIL }}
        run: |
          git config --global user.name "$X_USER_NAME"
          git config --global user.email "$X_USER_EMAIL"
          
          git lfs install
          git lfs track "mlruns/**"
          git add .gitattributes
          
          git add -f mlruns/
          git commit -m "Save mlruns from CI run [skip ci]" || true
          
          git remote set-url origin https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git
          git push origin HEAD:main