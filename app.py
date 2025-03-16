import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pickle

# Print TensorFlow version for debugging
print(f"TensorFlow version: {tf.__version__}")

class ThermoChauffageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Thermochauffage - Projet")
        
        # Variables pour le formulaire
        self.masse = tk.StringVar()
        self.volume = tk.StringVar()
        self.surface = tk.StringVar()
        self.materiau = tk.StringVar()
        self.file_path = None
        self.model = None
        self.scaler = None
        
        # Chargement du modèle pré-entraîné avec gestion d'erreur et tentative multiple
        try:
            # Try HDF5 format first
            self.model = tf.keras.models.load_model("thermochauffage_model.h5")
            print("Modèle chargé avec succès (HDF5).")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle (HDF5) : {e}")
            try:
                # Try SavedModel format as fallback
                self.model = tf.keras.models.load_model("thermochauffage_model")
                print("Modèle chargé avec succès (SavedModel).")
            except Exception as e2:
                print(f"Erreur lors du chargement du modèle (SavedModel) : {e2}")
                messagebox.showwarning("Avertissement", f"Impossible de charger le modèle : {e}\nAssurez-vous que 'thermochauffage_model.h5' ou 'thermochauffage_model' existe et est compatible avec TensorFlow {tf.__version__}.\nL'application continuera, mais les prédictions ne seront pas possibles.")
                self.model = None

        # Chargement du scaler avec gestion d'erreur
        try:
            with open("scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            print("Scaler chargé avec succès.")
        except Exception as e:
            print(f"Erreur lors du chargement du scaler : {e}")
            messagebox.showwarning("Avertissement", f"Impossible de charger le scaler : {e}\nAssurez-vous que 'scaler.pkl' existe.\nL'application continuera, mais les prédictions ne seront pas possibles.")
            self.scaler = None

        # Vérification si l'application peut fonctionner
        if self.model is None or self.scaler is None:
            messagebox.showinfo("Info", "L'application est en mode limité : exécutez le script d'entraînement pour générer 'thermochauffage_model.h5' ou 'thermochauffage_model' et 'scaler.pkl'.")
        
        # Création de l'interface
        self.create_widgets()

    def create_widgets(self):
        # Frame pour upload fichier
        upload_frame = tk.LabelFrame(self.root, text="Upload Fichier Solid", padx=10, pady=10)
        upload_frame.pack(padx=10, pady=5, fill="x")
        
        tk.Button(upload_frame, text="Choisir fichier", command=self.upload_file).pack()
        self.file_label = tk.Label(upload_frame, text="Aucun fichier sélectionné")
        self.file_label.pack()

        # Frame pour formulaire
        form_frame = tk.LabelFrame(self.root, text="Formulaire", padx=10, pady=10)
        form_frame.pack(padx=10, pady=5, fill="x")
        
        tk.Label(form_frame, text="Masse (kg, 0.1-2.0):").grid(row=0, column=0, pady=5)
        tk.Entry(form_frame, textvariable=self.masse).grid(row=0, column=1)
        
        tk.Label(form_frame, text="Volume (cm³, 20-200):").grid(row=1, column=0, pady=5)
        tk.Entry(form_frame, textvariable=self.volume).grid(row=1, column=1)
        
        tk.Label(form_frame, text="Surface (cm², 10-100):").grid(row=2, column=0, pady=5)
        tk.Entry(form_frame, textvariable=self.surface).grid(row=2, column=1)
        
        tk.Label(form_frame, text="Matériau:").grid(row=3, column=0, pady=5)
        tk.Entry(form_frame, textvariable=self.materiau).grid(row=3, column=1)

        # Bouton pour calculer, désactivé si modèle ou scaler manquant
        self.calc_button = tk.Button(form_frame, text="Calculer", command=self.calculate, state=tk.DISABLED if self.model is None or self.scaler is None else tk.NORMAL)
        self.calc_button.grid(row=4, column=0, columnspan=2, pady=10)

        # Frame pour la grille des résistances
        grid_frame = tk.LabelFrame(self.root, text="Grille des Résistances (40x20 cm)", padx=10, pady=10)
        grid_frame.pack(padx=10, pady=5, fill="x")
        
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=grid_frame)
        self.canvas.get_tk_widget().pack()
        self.update_grid(np.zeros((5, 5)))  # Grille initiale vide

    def upload_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Fichiers Solid", "*.sldprt *.step *.iges"), ("Tous", "*.*")])
        if self.file_path:
            self.file_label.config(text=f"Fichier: {self.file_path.split('/')[-1]}")
            messagebox.showinfo("Info", "Fichier uploadé. Extraction automatique non implémentée pour l'instant.")

    def update_grid(self, temperatures):
        self.ax.clear()
        im = self.ax.imshow(temperatures, cmap='hot', interpolation='nearest', vmin=50, vmax=300)
        self.fig.colorbar(im, ax=self.ax, label='Température (°C)')
        
        # Ajout des étiquettes de température sur chaque cellule
        for i in range(5):
            for j in range(5):
                self.ax.text(j, i, f"{temperatures[i, j]:.1f}" if self.model else "N/A", ha='center', va='center', color='black')
        
        self.ax.set_title("Disposition des 25 résistances")
        self.ax.set_xlabel("X (cm)")
        self.ax.set_ylabel("Y (cm)")
        self.ax.set_xticks(np.arange(5))
        self.ax.set_yticks(np.arange(5))
        self.ax.set_xticklabels(np.arange(0, 40, 8))  # Échelle de 0 à 40 cm
        self.ax.set_yticklabels(np.arange(0, 20, 4))  # Échelle de 0 à 20 cm
        self.canvas.draw()

    def calculate(self):
        if self.model is None or self.scaler is None:
            messagebox.showerror("Erreur", "Modèle ou scaler non chargé. Exécutez le script d'entraînement d'abord.")
            return

        try:
            # Récupérer les valeurs du formulaire
            masse = float(self.masse.get())
            volume = float(self.volume.get())
            surface = float(self.surface.get())
            materiau = self.materiau.get()

            # Validation des plages de valeurs (basée sur les données d'entraînement)
            if not (0.1 <= masse <= 2.0):
                raise ValueError("La masse doit être entre 0.1 et 2.0 kg.")
            if not (20 <= volume <= 200):
                raise ValueError("Le volume doit être entre 20 et 200 cm³.")
            if not (10 <= surface <= 100):
                raise ValueError("La surface doit être entre 10 et 100 cm².")

            # Calcul du nombre de pièces
            total_surface = 800  # 40x20 cm
            nb_pieces = int(total_surface / surface)
            messagebox.showinfo("Résultat", f"Nombre de pièces possibles : {nb_pieces}")

            # Prédiction des températures avec le modèle chargé
            puissance = 1026  # 380V * 2.7A
            input_data = np.array([[puissance, masse, surface, volume]])
            # Normalisation des données d'entrée avec le scaler
            input_data_scaled = self.scaler.transform(input_data)
            temp_predite = self.model.predict(input_data_scaled, verbose=0)[0][0]

            # Ajuster la grille avec une variation simple (exemple)
            temperatures = np.full((5, 5), temp_predite)
            for i in range(5):
                for j in range(5):
                    temperatures[i, j] += np.random.normal(0, 5)  # Ajout de bruit pour variation
            temperatures = np.clip(temperatures, 50, 300)  # Limiter entre 50 et 300°C
            self.update_grid(temperatures)

            messagebox.showinfo("Température", f"Température moyenne prédite par résistance : {temp_predite:.2f} °C")

        except ValueError as e:
            messagebox.showerror("Erreur", str(e))
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur s'est produite : {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ThermoChauffageApp(root)
    root.mainloop()