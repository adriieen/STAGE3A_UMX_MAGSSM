import torch

# 1. Création d'un tenseur contigu (2 lignes, 3 colonnes)
# Mémoire physique : [1, 2, 3, 4, 5, 6]
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Original contigu : {x.is_contiguous()}") 

# 2. Transposition
# Les métadonnées changent, mais la mémoire physique reste : [1, 2, 3, 4, 5, 6]
# Pour lire la première colonne [1, 4], PyTorch doit "sauter" 3 cases.
x_t = x.transpose(0, 1)
print(f"Après transpose, contigu : {x_t.is_contiguous()}")

# 3. Tentative de .view() -> Erreur
# PyTorch refuse car il ne peut pas garantir une lecture linéaire simple 
# sur un stockage qui n'est plus aligné avec la structure logique.
try:
    échec = x_t.view(-1)
except RuntimeError as e:
    print(f"Erreur attendue : {e}")

# 4. Passage par .contiguous()
# PyTorch alloue un nouveau bloc mémoire et y range les données dans l'ordre de lecture actuel :
# Nouvelle mémoire physique : [1, 4, 2, 5, 3, 6]
x_contig = x_t.contiguous()
print(f"Après .contiguous(), contigu : {x_contig.is_contiguous()}")

# 5. Le .view() fonctionne enfin
# Comme la mémoire physique est alignée, la vue est possible sans copie supplémentaire.
succès = x_contig.view(2, 3)
print("Résultat final du view après contiguous :")
print(succès)