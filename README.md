# Restauration d'images par Plug-and-Play (GS-DRUNet)

Projet semestriel M2 Data Science & Vision par ordinateur — Université de Bordeaux.

- `PROJET_IMAGE_4.ipynb` : le notebook du projet (expériences complètes).
- `app.py` : application Streamlit interactive (défloutage, inpainting, IRM).
- `pnp_core.py` : les algorithmes du notebook, réorganisés en fonctions réutilisables par l'app.
- `assets/` : images d'exemple et figures de résultats extraites du notebook.

## Lancer l'application
```bash
conda create -n pnp python=3.11 -y
conda activate pnp
pip install -r requirements.txt
streamlit run app.py
```
Au premier lancement, les poids de GS-DRUNet (~70 Mo) sont téléchargés depuis Hugging Face.
