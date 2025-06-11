import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import streamlit as st
import torch
import numpy as np
import open_clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# --- Chargement modèle CLIP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model = model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# --- Chargement des embeddings ---
EMBEDDINGS_PATH = "embeddings"
image_paths = np.load(f"{EMBEDDINGS_PATH}/filenames.npy", allow_pickle=True)
image_embeddings = np.load(f"{EMBEDDINGS_PATH}/image_embeddings.npy")

# --- Titre de l'app ---
st.title("Moteur de recherche d'art avec CLIP")

# --- Entrée utilisateur ---
query = st.text_input("Entrez votre requête (ex : femme au chapeau, paysage nocturne...)", "")

if query:
    with st.spinner("Recherche en cours..."):
        # Encodage du texte
        tokens = tokenizer([query]).to(device)
        with torch.no_grad():
            text_embedding = model.encode_text(tokens).cpu().numpy()

        # Calcul similarité cosinus
        sims = cosine_similarity(text_embedding, image_embeddings)[0]
        top_indices = sims.argsort()[::-1][:5]

        # Affichage
        st.subheader("Résultats les plus proches :")
        for idx in top_indices:
            score = sims[idx]
            img = Image.open(image_paths[idx])
            st.image(img, caption=f"{image_paths[idx]} — Sim : {score:.3f}", use_container_width=True)
        
            # --- Section "Images similaires" ---
            st.markdown("**Images similaires :**")
            img_embedding = image_embeddings[idx].reshape(1, -1)
            sim_to_others = cosine_similarity(img_embedding, image_embeddings)[0]
            similar_indices = sim_to_others.argsort()[::-1][1:4]
        
            cols = st.columns(3)
            for col, sim_idx in zip(cols, similar_indices):
                similar_img = Image.open(image_paths[sim_idx])
                with col:
                    st.image(similar_img, caption=f"Sim : {sim_to_others[sim_idx]:.3f}", use_container_width=True)
