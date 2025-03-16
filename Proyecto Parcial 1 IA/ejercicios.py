import flet as ft
import random
import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools

# **Base de Datos de Ejercicios**
ejercicios = {
    "Pecho": ["Press de banca", "Aperturas", "Flexiones", "Press inclinado"],
    "Espalda": ["Dominadas", "Remo con barra", "Jalón al pecho"],
    "Piernas": ["Sentadilla", "Peso muerto", "Prensa", "Desplantes"],
    "Hombros": ["Press militar", "Elevaciones laterales"],
    "Bíceps": ["Curl con barra", "Martillo"],
    "Tríceps": ["Fondos", "Press francés"],
    "Abdomen": ["Plancha", "Crunches"]
}

# **Parámetros según el objetivo**
objetivos = {
    "Hipertrofia": {"reps": (6, 12), "series": (4, 5)},
    "Resistencia": {"reps": (12, 20), "series": (3, 4)},
    "Fuerza": {"reps": (3, 6), "series": (4, 6)}
}

# 🔹 Configuración del Algoritmo Genético
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def generar_individuo():
    return [random.choice(ejercicios[musculo]) for musculo in ejercicios]

def evaluar(individuo, objetivo):
    """Evalúa la rutina basándose en el objetivo del usuario."""
    obj_config = objetivos[objetivo]
    rep_total = sum(random.randint(*obj_config["reps"]) for _ in individuo)
    series_total = sum(random.randint(*obj_config["series"]) for _ in individuo)
    
    # 🔹 Evaluación: Máxima diversidad + ajuste al objetivo
    fitness = len(set(individuo)) + (rep_total / 100) + (series_total / 20)
    return (fitness,)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, generar_individuo)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.3)  
toolbox.register("select", tools.selTournament, tournsize=3)

# **Variable global para almacenar la evolución del fitness**
fitness_evolucion = []

def evolucionar_rutina(objetivo):
    """Ejecuta el Algoritmo Genético y guarda la evolución del fitness basado en el objetivo."""
    global fitness_evolucion
    fitness_evolucion = []
    poblacion = toolbox.population(n=10)

    for gen in range(10):  
        hijos = list(map(toolbox.clone, poblacion))
        for child1, child2 in zip(hijos[::2], hijos[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values
        for mutante in hijos:
            if random.random() < 0.5:
                toolbox.mutate(mutante)
                del mutante.fitness.values
        poblacion.extend(hijos)
        
        # **Evaluar todos los individuos con el objetivo seleccionado**
        fits = [evaluar(ind, objetivo) for ind in poblacion]
        for ind, fit in zip(poblacion, fits):
            ind.fitness.values = fit
        
        # **Seleccionar la mejor rutina**
        poblacion = toolbox.select(poblacion, k=10)
        max_fitness = max([ind.fitness.values[0] for ind in poblacion])
        fitness_evolucion.append(max_fitness)

    return poblacion[0]  

def mostrar_grafica_matplot():
    """Muestra la gráfica con la evolución del fitness en la rutina generada."""
    if not fitness_evolucion:
        print("⚠️ No hay gráfica generada, primero presiona 'Generar Rutina'.")
        return

    generaciones = list(range(1, 11))
    plt.figure(figsize=(8, 5))
    plt.plot(generaciones, fitness_evolucion, marker='o', linestyle='-', color='blue', label="Mejor Fitness")

    mejor_gen = np.argmax(fitness_evolucion) + 1
    mejor_valor = np.max(fitness_evolucion)
    plt.scatter(mejor_gen, mejor_valor, color='red', s=100, label=f"Mejor Gen ({mejor_gen})")
    plt.text(mejor_gen, mejor_valor + 0.5, f"{mejor_valor:.2f}", color='red', fontsize=12)

    plt.xlabel("Generación")
    plt.ylabel("Fitness (Diversidad + Ajuste al Objetivo)")
    plt.title("Evolución del Fitness - Algoritmo Genético")
    plt.legend()
    plt.grid()
    plt.show()

# **Interfaz con Flet**
def main(page: ft.Page):
    page.title = "Generador de Rutinas con Algoritmos Genéticos"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    objetivo_dropdown = ft.Dropdown(
        label="Objetivo", 
        options=[ft.dropdown.Option(o) for o in objetivos.keys()], 
        width=200
    )

    dias_dropdown = ft.Dropdown(
        label="Días por semana", 
        options=[ft.dropdown.Option(str(i)) for i in [3, 4, 5, 6]], 
        width=200
    )

    tabla = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Día")),
            ft.DataColumn(ft.Text("Ejercicio")),
            ft.DataColumn(ft.Text("Series")),
            ft.DataColumn(ft.Text("Reps")),
            ft.DataColumn(ft.Text("Descanso"))
        ],
        rows=[]
    )

    tabla_scroll = ft.Container(
        content=ft.Column([tabla], scroll=ft.ScrollMode.AUTO), 
        height=400, 
        expand=True
    )

    def generar_rutina_flet(e):
        if not objetivo_dropdown.value or not dias_dropdown.value:
            return
        
        dias = int(dias_dropdown.value)
        objetivo = objetivo_dropdown.value

        mejor_rutina = evolucionar_rutina(objetivo)

        tabla.rows.clear()
        for dia in range(1, dias + 1):
            for ejercicio in mejor_rutina[:4]:  
                tabla.rows.append(ft.DataRow(cells=[
                    ft.DataCell(ft.Text(f"Día {dia}")),
                    ft.DataCell(ft.Text(ejercicio)),  
                    ft.DataCell(ft.Text(str(random.randint(*objetivos[objetivo]["series"])))),
                    ft.DataCell(ft.Text(str(random.randint(*objetivos[objetivo]["reps"])))),
                    ft.DataCell(ft.Text("60-90 seg"))
                ]))

        page.update()

    def mostrar_grafica(e):
        """Llama a la función que muestra la gráfica con Matplotlib."""
        mostrar_grafica_matplot()

    generar_rutina_boton = ft.ElevatedButton("Generar Rutina", on_click=generar_rutina_flet, width=200, height=50)
    mostrar_grafica_boton = ft.ElevatedButton("Mostrar Gráfica", on_click=mostrar_grafica, width=200, height=50)

    layout = ft.Row([
        ft.Column([objetivo_dropdown, dias_dropdown, generar_rutina_boton, mostrar_grafica_boton], spacing=20),
        tabla_scroll  
    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)  

    page.add(layout)

ft.app(target=main)
