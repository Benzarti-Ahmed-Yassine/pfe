import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import numpy as np
import tensorflow as tf
import pickle
import os

# Print TensorFlow version for debugging
print(f"TensorFlow version: {tf.__version__}")

class ThermoChauffageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Thermochauffage - Projet (Multi-Moule)")
        
        # Adjust paths relative to the executable location
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(__file__)
        
        model_path = os.path.join(base_path, "thermochauffage_model.h5")
        scaler_path = os.path.join(base_path, "scaler.pkl")

        # Variables globales
        self.model = None
        self.scaler = None
        self.molds = []  # List to store mold configurations
        self.nb_molds_var = tk.StringVar(value="1")  # Number of molds to configure
        
        # Load the pre-trained model with proper custom objects
        try:
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
            )
            print("Modèle chargé avec succès (HDF5).")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle (HDF5) : {e}")
            messagebox.showwarning(
                "Avertissement",
                f"Impossible de charger le modèle : {e}\nAssurez-vous que 'thermochauffage_model.h5' existe dans le répertoire de l'application."
            )
            self.model = None

        # Load the scaler with error handling
        try:
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            print("Scaler chargé avec succès.")
        except Exception as e:
            print(f"Erreur lors du chargement du scaler : {e}")
            messagebox.showwarning(
                "Avertissement",
                f"Impossible de charger le scaler : {e}\nAssurez-vous que 'scaler.pkl' existe dans le répertoire de l'application."
            )
            self.scaler = None

        # Check if the app can function
        if self.model is None or self.scaler is None:
            messagebox.showinfo(
                "Info",
                "L'application est en mode limité : exécutez le script d'entraînement pour générer 'thermochauffage_model.h5' et 'scaler.pkl'."
            )
        
        # Create the interface
        self.create_widgets()

    def create_widgets(self):
        # Frame for number of molds
        nb_molds_frame = tk.LabelFrame(self.root, text="Nombre de Moules", padx=10, pady=10)
        nb_molds_frame.pack(padx=10, pady=5, fill="x")
        
        tk.Label(nb_molds_frame, text="Nombre de moules à configurer :").grid(row=0, column=0, pady=5)
        tk.Entry(nb_molds_frame, textvariable=self.nb_molds_var, width=5).grid(row=0, column=1)
        tk.Button(nb_molds_frame, text="Configurer", command=self.setup_molds).grid(row=0, column=2, padx=5)

        # Frame for mold forms (will be populated dynamically)
        self.molds_frame = tk.LabelFrame(self.root, text="Configuration des Moules", padx=10, pady=10)
        self.molds_frame.pack(padx=10, pady=5, fill="x")

        # Frame for resistance grid
        grid_frame = tk.LabelFrame(self.root, text="Grille des Résistances (40x20 cm)", padx=10, pady=10)
        grid_frame.pack(padx=10, pady=5, fill="both", expand=True)
        
        self.tree = ttk.Treeview(grid_frame, columns=("Y0", "Y4", "Y8", "Y12", "Y16"), show="headings", height=5)
        self.tree.pack(fill="both", expand=True)
        
        for col in ("Y0", "Y4", "Y8", "Y12", "Y16"):
            self.tree.heading(col, text=f"Y={col[1:]} cm")
            self.tree.column(col, width=100, anchor="center")
        
        self.update_table(np.zeros((5, 5)))

        # Frame for results
        self.result_frame = tk.LabelFrame(self.root, text="Résultats", padx=10, pady=10)
        self.result_frame.pack(padx=10, pady=5, fill="x")
        self.result_text = tk.Text(self.result_frame, height=10, width=50)
        self.result_text.pack()

    def setup_molds(self):
        try:
            nb_molds = int(self.nb_molds_var.get())
            if nb_molds < 1 or nb_molds > 5:  # Arbitrary limit to 5 molds for simplicity
                raise ValueError("Le nombre de moules doit être entre 1 et 5.")
            
            # Clear previous mold forms
            for widget in self.molds_frame.winfo_children():
                widget.destroy()
            self.molds = []

            # Create a form for each mold
            for i in range(nb_molds):
                mold_vars = {
                    "masse_moule": tk.StringVar(),
                    "volume_moule": tk.StringVar(),
                    "surface_moule": tk.StringVar(),
                    "materiau_moule": tk.StringVar(),
                    "masse_contre_moule": tk.StringVar(),
                    "volume_contre_moule": tk.StringVar(),
                    "surface_contre_moule": tk.StringVar(),
                    "materiau_contre_moule": tk.StringVar(),
                    "materiau_moulage": tk.StringVar(),
                    "file_path": None
                }
                self.molds.append(mold_vars)

                mold_frame = tk.LabelFrame(self.molds_frame, text=f"Moule {i+1}", padx=5, pady=5)
                mold_frame.grid(row=i//2, column=i%2, padx=5, pady=5, sticky="nsew")

                # Mold properties
                tk.Label(mold_frame, text="Moule - Masse (kg, 0.1-2.0):").grid(row=0, column=0, pady=2)
                tk.Entry(mold_frame, textvariable=mold_vars["masse_moule"]).grid(row=0, column=1)
                
                tk.Label(mold_frame, text="Moule - Volume (cm³, 20-200):").grid(row=1, column=0, pady=2)
                tk.Entry(mold_frame, textvariable=mold_vars["volume_moule"]).grid(row=1, column=1)
                
                tk.Label(mold_frame, text="Moule - Surface (cm², 10-100):").grid(row=2, column=0, pady=2)
                tk.Entry(mold_frame, textvariable=mold_vars["surface_moule"]).grid(row=2, column=1)
                
                tk.Label(mold_frame, text="Moule - Matériau:").grid(row=3, column=0, pady=2)
                tk.Entry(mold_frame, textvariable=mold_vars["materiau_moule"]).grid(row=3, column=1)

                # Counter-mold properties
                tk.Label(mold_frame, text="Contre-Moule - Masse (kg, 0.1-2.0):").grid(row=4, column=0, pady=2)
                tk.Entry(mold_frame, textvariable=mold_vars["masse_contre_moule"]).grid(row=4, column=1)
                
                tk.Label(mold_frame, text="Contre-Moule - Volume (cm³, 20-200):").grid(row=5, column=0, pady=2)
                tk.Entry(mold_frame, textvariable=mold_vars["volume_contre_moule"]).grid(row=5, column=1)
                
                tk.Label(mold_frame, text="Contre-Moule - Surface (cm², 10-100):").grid(row=6, column=0, pady=2)
                tk.Entry(mold_frame, textvariable=mold_vars["surface_contre_moule"]).grid(row=6, column=1)
                
                tk.Label(mold_frame, text="Contre-Moule - Matériau:").grid(row=7, column=0, pady=2)
                tk.Entry(mold_frame, textvariable=mold_vars["materiau_contre_moule"]).grid(row=7, column=1)

                # Molding material
                tk.Label(mold_frame, text="Matériau de Moulage:").grid(row=8, column=0, pady=2)
                tk.Entry(mold_frame, textvariable=mold_vars["materiau_moulage"]).grid(row=8, column=1)

                # File upload
                tk.Button(mold_frame, text="Charger fichier", command=lambda idx=i: self.upload_file(idx)).grid(row=9, column=0, columnspan=2, pady=5)

            # Add a calculate button
            tk.Button(self.molds_frame, text="Calculer", command=self.calculate).grid(row=nb_molds//2 + 1, column=0, columnspan=2, pady=10)

        except ValueError as e:
            messagebox.showerror("Erreur", str(e))

    def upload_file(self, mold_idx):
        file_path = filedialog.askopenfilename(filetypes=[("Fichiers Solid", "*.sldprt *.step *.iges"), ("Tous", "*.*")])
        if file_path:
            self.molds[mold_idx]["file_path"] = file_path
            messagebox.showinfo("Info", f"Fichier pour Moule {mold_idx+1} : {file_path.split('/')[-1]}")

    def update_table(self, temperatures):
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for i in range(5):
            row_values = [f"{temperatures[i, j]:.1f}" if self.model else "N/A" for j in range(5)]
            self.tree.insert("", "end", values=row_values, tags=(f"row{i}",))
        
        self.tree["displaycolumns"] = ("Y0", "Y4", "Y8", "Y12", "Y16")
        self.tree.tag_configure("row0", background="#f0f0f0")
        self.tree.tag_configure("row1", background="#ffffff")
        self.tree.tag_configure("row2", background="#f0f0f0")
        self.tree.tag_configure("row3", background="#ffffff")
        self.tree.tag_configure("row4", background="#f0f0f0")

    def calculate(self):
        if self.model is None or self.scaler is None:
            messagebox.showerror("Erreur", "Modèle ou scaler non chargé. Exécutez le script d'entraînement d'abord.")
            return

        self.result_text.delete(1.0, tk.END)
        total_surface = 4096  # 40x20 cm
        puissance = 1026  # 380V * 2.7A
        grid_temperatures = np.zeros((5, 5))  # 5x5 grid for temperatures
        mold_positions = []

        for idx, mold in enumerate(self.molds):
            try:
                # Retrieve mold and counter-mold properties
                masse_moule = float(mold["masse_moule"].get())
                volume_moule = float(mold["volume_moule"].get())
                surface_moule = float(mold["surface_moule"].get())
                materiau_moule = mold["materiau_moule"].get()

                masse_contre_moule = float(mold["masse_contre_moule"].get())
                volume_contre_moule = float(mold["volume_contre_moule"].get())
                surface_contre_moule = float(mold["surface_contre_moule"].get())
                materiau_contre_moule = mold["materiau_contre_moule"].get()

                materiau_moulage = mold["materiau_moulage"].get()

                # Validation for mold
                if not (0.1 <= masse_moule <= 2.0):
                    raise ValueError(f"Moule {idx+1} : La masse du moule doit être entre 0.1 et 2.0 kg.")
                if not (20 <= volume_moule <= 200):
                    raise ValueError(f"Moule {idx+1} : Le volume du moule doit être entre 20 et 200 cm³.")
                if not (10 <= surface_moule <= 100):
                    raise ValueError(f"Moule {idx+1} : La surface du moule doit être entre 10 et 100 cm².")

                # Validation for counter-mold
                if not (0.1 <= masse_contre_moule <= 2.0):
                    raise ValueError(f"Moule {idx+1} : La masse du contre-moule doit être entre 0.1 et 2.0 kg.")
                if not (20 <= volume_contre_moule <= 200):
                    raise ValueError(f"Moule {idx+1} : Le volume du contre-moule doit être entre 20 et 200 cm³.")
                if not (10 <= surface_contre_moule <= 100):
                    raise ValueError(f"Moule {idx+1} : La surface du contre-moule doit être entre 10 et 100 cm².")

                # Calculate number of pieces (using the mold's surface)
                nb_pieces = int(total_surface / surface_moule)

                # Assign position on the grid (simplified: divide the grid into regions)
                x_pos = (idx % 2) * 20  # X position (0 or 20 cm)
                y_pos = (idx // 2) * 10  # Y position (0, 10, 20, ...)
                mold_positions.append((x_pos, y_pos))

                # Predict temperature for the mold (simplified: use mold properties for prediction)
                input_data = np.array([[puissance, masse_moule, surface_moule, volume_moule]])
                input_data_scaled = self.scaler.transform(input_data)
                temp_predite = self.model.predict(input_data_scaled, verbose=0)[0][0]

                # Adjust temperature based on molding material (simplified adjustment)
                thermal_factor = 1.0
                if materiau_moulage.lower() in ["plastique", "plastic"]:
                    thermal_factor = 0.9  # Plastics typically conduct less heat
                elif materiau_moulage.lower() in ["métal", "metal"]:
                    thermal_factor = 1.1  # Metals conduct more heat
                temp_predite *= thermal_factor

                # Update grid temperatures for the mold's position
                x_idx = (x_pos // 8)  # Map X position to grid (0-4)
                y_idx = (y_pos // 4)  # Map Y position to grid (0-4)
                grid_temperatures[x_idx, y_idx] = temp_predite

                # Display results for this mold
                self.result_text.insert(tk.END, f"Moule {idx+1} :\n")
                self.result_text.insert(tk.END, f"  Position : X={x_pos} cm, Y={y_pos} cm\n")
                self.result_text.insert(tk.END, f"  Nombre de pièces possibles : {nb_pieces}\n")
                self.result_text.insert(tk.END, f"  Matériau du moule : {materiau_moule}\n")
                self.result_text.insert(tk.END, f"  Matériau du contre-moule : {materiau_contre_moule}\n")
                self.result_text.insert(tk.END, f"  Matériau de moulage : {materiau_moulage}\n")
                self.result_text.insert(tk.END, f"  Température prédite : {temp_predite:.2f} °C\n\n")

            except ValueError as e:
                messagebox.showerror("Erreur", str(e))
                return
            except Exception as e:
                messagebox.showerror("Erreur", f"Moule {idx+1} : Une erreur s'est produite : {e}")
                return

        # Update the grid with the combined temperatures
        for i in range(5):
            for j in range(5):
                if grid_temperatures[i, j] == 0:  # If no mold affects this cell
                    grid_temperatures[i, j] = 50  # Default temperature
                else:
                    grid_temperatures[i, j] += np.random.normal(0, 5)  # Add noise for variation
        grid_temperatures = np.clip(grid_temperatures, 50, 300)  # Limit between 50 and 300°C
        self.update_table(grid_temperatures)

if __name__ == "__main__":
    import sys
    root = tk.Tk()
    app = ThermoChauffageApp(root)
    root.mainloop()