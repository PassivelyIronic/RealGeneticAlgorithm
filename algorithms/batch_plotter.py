import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np

class ResultsWindow:
    def __init__(self, results):
        self.results = pd.DataFrame(results)
        self.root = tk.Tk()
        self.root.title("Porównanie algorytmów GA i GJO")
        self.root.geometry("1200x800")

        self.create_widgets()

    def create_widgets(self):
        # Frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Label(control_frame, text="Wybierz rozmiar populacji:").pack(side=tk.LEFT)
        self.pop_size_var = tk.StringVar()
        pop_sizes = sorted(self.results["population_size"].unique())
        self.pop_size_var.set(str(pop_sizes[0]))
        pop_menu = tk.OptionMenu(control_frame, self.pop_size_var, *map(str, pop_sizes), command=self.update_plots)
        pop_menu.pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Wybierz liczbę epok:").pack(side=tk.LEFT, padx=10)
        self.epochs_var = tk.StringVar()
        epochs = sorted(self.results["epochs"].unique())
        self.epochs_var.set(str(epochs[0]))
        epochs_menu = tk.OptionMenu(control_frame, self.epochs_var, *map(str, epochs), command=self.update_plots)
        epochs_menu.pack(side=tk.LEFT, padx=5)

        # Frame for results text
        self.results_text = tk.Text(self.root, height=10, width=140)
        self.results_text.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Frame for plots
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_plots()

    def update_plots(self, *args):
        pop_size = int(self.pop_size_var.get())
        epochs = int(self.epochs_var.get())

        # Filter results for selected population size and epochs
        filtered = self.results[(self.results["population_size"] == pop_size) & (self.results["epochs"] == epochs)]

        # Clear previous plot
        self.ax.clear()

        # Plot best fitness history for GA and GJO
        for alg in ["GA", "GJO"]:
            alg_data = filtered[filtered["algorithm"] == alg]
            if alg_data.empty:
                continue
            # Jeśli masz best_fitness_history jako listę w wynikach, użyj jej:
            if "best_fitness_history" in alg_data.columns and not alg_data["best_fitness_history"].isnull().all():
                # Zakładamy, że każda próba ma best_fitness_history jako listę
                histories = alg_data["best_fitness_history"].dropna().tolist()
                # Uśrednij historie po próbach
                max_len = max(len(h) for h in histories)
                avg_history = np.mean([np.pad(h, (0, max_len - len(h)), 'edge') for h in histories], axis=0)
                self.ax.plot(range(1, len(avg_history)+1), avg_history, label=f"{alg} (pop={pop_size})")
            else:
                # Jeśli nie ma historii, zasymuluj jako linię stałą
                best_fitness = alg_data["best_fitness"].mean()
                self.ax.plot(range(1, epochs+1), [best_fitness]*epochs, label=f"{alg} (pop={pop_size})")

        self.ax.set_title(f"Historia wartości funkcji celu - populacja: {pop_size}, epoki: {epochs}")
        self.ax.set_xlabel("Iteracja")
        self.ax.set_ylabel("Wartość funkcji celu")
        self.ax.legend()
        self.ax.grid(True)

        self.canvas.draw()

        # Update results text
        self.results_text.delete(1.0, tk.END)
        for alg in ["GA", "GJO"]:
            alg_data = filtered[filtered["algorithm"] == alg]
            if alg_data.empty:
                continue
            best_fit = alg_data["best_fitness"].mean()
            exec_time = alg_data["execution_time"].mean()
            self.results_text.insert(tk.END, f"{alg} - Najlepsze przystosowanie: {best_fit:.4f}, Czas wykonania: {exec_time:.2f} s\n")

    def run(self):
        self.root.mainloop()


def show_results_window(results):
    window = ResultsWindow(results)
    window.run()