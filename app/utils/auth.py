# ============================================================
# utils/auth.py — Authentification et contrôle d'accès (RBAC)
# ============================================================
# Couche de sécurité de l'application :
# - Authentification via streamlit-authenticator + bcrypt
# - RBAC avec 3 rôles : admin, gestionnaire, lecteur
# - Fonctions utilitaires pour les guards de pages
# ============================================================

import streamlit as st
import yaml
import os
import bcrypt
import streamlit_authenticator as stauth

# ============================================================
# Chemins
# ============================================================
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


# ============================================================
# Chargement / sauvegarde de la configuration
# ============================================================
def charger_config():
    """Charge le fichier config.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sauvegarder_config(config):
    """Sauvegarde le fichier config.yaml (utilisé par l'admin)."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


# ============================================================
# Authentification
# ============================================================
def initialiser_authentification():
    """
    Initialise et affiche le formulaire de login.
    Retourne l'objet authenticator et le config.
    """
    config = charger_config()

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )

    return authenticator, config


def get_role_utilisateur(config):
    """
    Retourne le rôle de l'utilisateur connecté.
    Doit être appelé APRÈS l'authentification réussie.
    """
    username = st.session_state.get("username", None)
    if username and username in config["credentials"]["usernames"]:
        return config["credentials"]["usernames"][username].get("role", "lecteur")
    return "lecteur"


# ============================================================
# RBAC — Contrôle d'accès par rôle
# ============================================================
# Hiérarchie des rôles :
#   admin > gestionnaire > lecteur
#
# Accès par module :
#   Module 1 (Visualisation)  → lecteur, gestionnaire, admin
#   Module 2 (Prédiction)     → gestionnaire, admin
#   Module 3 (Optimisation)   → gestionnaire, admin
#   Admin (gestion users)     → admin uniquement

ROLES_AUTORISES = {
    "visualisation": ["lecteur", "gestionnaire", "admin"],
    "prediction": ["gestionnaire", "admin"],
    "optimisation": ["gestionnaire", "admin"],
    "admin": ["admin"],
}


def verifier_acces(module: str) -> bool:
    """
    Vérifie si l'utilisateur courant a accès au module demandé.
    
    Args:
        module: clé du module ('visualisation', 'prediction', 'optimisation', 'admin')
    
    Returns:
        True si accès autorisé, False sinon.
    """
    role = st.session_state.get("role", "lecteur")
    roles_ok = ROLES_AUTORISES.get(module, [])
    return role in roles_ok


def guard_page(module: str):
    """
    Bloque l'accès à une page si le rôle est insuffisant.
    Affiche un message d'erreur et stoppe l'exécution.
    
    Utilisation dans chaque page :
        from utils.auth import guard_page
        guard_page("prediction")  # En haut du fichier, après le check des données
    """
    if not verifier_acces(module):
        role = st.session_state.get("role", "lecteur")
        st.error(
            f"🔒 **Accès refusé** — Votre rôle ({role}) ne permet pas "
            f"d'accéder à ce module.\n\n"
            f"Rôles autorisés : {', '.join(ROLES_AUTORISES.get(module, []))}"
        )
        st.page_link("main.py", label="← Retour à l'accueil", icon="🏠")
        st.stop()


# ============================================================
# Admin — Gestion des utilisateurs
# ============================================================
def afficher_panneau_admin(config):
    """
    Affiche le panneau d'administration dans un expander.
    Permet à l'admin de voir, ajouter et supprimer des utilisateurs.
    """
    if not verifier_acces("admin"):
        return

    with st.expander("🔧 Administration - Gestion des utilisateurs"):
        st.markdown("### Utilisateurs actuels")

        users = config["credentials"]["usernames"]
        users_data = []
        for username, info in users.items():
            users_data.append({
                "Identifiant": username,
                "Nom": info.get("name", ""),
                "Rôle": info.get("role", "lecteur"),
            })

        import pandas as pd
        df_users = pd.DataFrame(users_data)
        st.dataframe(df_users, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Ajouter un utilisateur")

        new_username = st.text_input("Identifiant", key="new_username")
        new_name = st.text_input("Nom complet", key="new_name")
        new_password = st.text_input("Mot de passe", type="password", key="new_password")
        new_role = st.selectbox("Rôle", ["lecteur", "gestionnaire", "admin"], key="new_role")

        if st.button("➕ Ajouter l'utilisateur", key="btn_add_user"):
            if not new_username or not new_password or not new_name:
                st.warning("Tous les champs sont obligatoires.")
            elif new_username in users:
                st.warning(f"L'identifiant '{new_username}' existe déjà.")
            else:
                # Hasher le mot de passe avec bcrypt
                hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                config["credentials"]["usernames"][new_username] = {
                    "name": new_name,
                    "password": hashed,
                    "role": new_role,
                }
                sauvegarder_config(config)
                st.success(f"✅ Utilisateur '{new_username}' ajouté avec le rôle '{new_role}'.")
                st.rerun()

        st.markdown("---")
        st.markdown("### Supprimer un utilisateur")

        # Ne pas permettre la suppression de son propre compte
        current_user = st.session_state.get("username", "")
        deletable = [u for u in users.keys() if u != current_user]

        if deletable:
            user_to_delete = st.selectbox("Utilisateur à supprimer", deletable, key="del_user")
            if st.button("🗑️ Supprimer", key="btn_del_user"):
                del config["credentials"]["usernames"][user_to_delete]
                sauvegarder_config(config)
                st.success(f"✅ Utilisateur '{user_to_delete}' supprimé.")
                st.rerun()
        else:
            st.info("Aucun autre utilisateur à supprimer.")
