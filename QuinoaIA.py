import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import random
from typing import List, Tuple, Callable

"""
quinoa.py - Modelo de Optimizacion para Produccion de Quinua en Puno, Peru

Este modulo implementa modelos matematicos y de machine learning para optimizar 
la produccion de quinua en la region de Puno, basado en investigacion de la 
Universidad Nacional del Altiplano.

Contiene:
1. Modelo de optimizacion de utilidad y punto de equilibrio
2. Algoritmo genetico para maximizacion de utilidad
3. Red neuronal para predecir utilidad
4. Modelo de minimizacion de costos

Autor: Universidad Nacional del Altiplano, Puno
Referencia: Articulo de investigacion sobre optimizacion de quinua (2023)
"""

class QuinoaOptimizationModel:
    """Modelo de optimizacion para produccion de quinua.
    
    Implementa las ecuaciones y restricciones del articulo de investigacion:
    - Maximizacion de utilidad (Eq. 4)
    - Calculo de punto de equilibrio (Eq. 5)
    - Minimizacion de costos (Eq. 6)
    - Restricciones del modelo (Seccion III)
    
    Parametros base:
    pp = 17.00         # Precio base de venta (S/.)
    production_per_hectare = 1200  # kg por hectarea
    cost_per_hectare = 13242.00    # Costo por hectarea (S/.)
    max_hectares = 100             # Limite maximo de terreno
    max_demand = 150               # Demanda maxima (toneladas?)
    competition_production = 0.25  # Produccion de competidores
    """
    
    def __init__(self):
        """Inicializa el modelo con parametros predeterminados."""
        self.pp = 17.00
        self.production_per_hectare = 1200
        self.cost_per_hectare = 13242.00
        self.max_hectares = 100
        self.max_demand = 150
        self.competition_production = 0.25
        
    def utility_maximization(self, x1: float, x2: float) -> float:
        """
        Calcula la utilidad (ganancia) de la produccion de quinua.
        Ecuacion (4) del articulo: Max Z = 1.2x₁x₂ - 13.2x₁
        
        Args:
            x1: Terreno cultivado (hectareas)
            x2: Precio de venta por kg (S/.)
            
        Returns:
            Utilidad en miles de S/.
        """
        return 1.2 * x1 * x2 - 13.2 * x1
    
    def check_constraints(self, x1: float, x2: float, x3: float, x4: float) -> bool:
        """
        Verifica si una solucion satisface todas las restricciones.
        Basado en la seccion III del articulo.
        
        Args:
            x1: Terreno cultivado (hectareas)
            x2: Precio de venta por kg (S/.)
            x3: Demanda del mercado actual
            x4: Produccion de competidores
            
        Returns:
            True si todas las restricciones se satisfacen, False en caso contrario.
        """
        # Restricciones de capacidad maxima
        if x1 > self.max_hectares or x3 > self.max_demand:
            return False
            
        # Restriccion de produccion de competidores
        if x4 != self.competition_production:
            return False
            
        # Restriccion de precio maximo
        price_constraint = 34 - (20.4 * x1 / x3) + 17 * x4
        if x2 > price_constraint:
            return False
            
        return True
    
    def calculate_break_even_price(self) -> float:
        """
        Calcula el precio de punto de equilibrio (utilidad = 0).
        Ecuacion (5) del articulo: 1.2x₂ ≥ 13.2
        
        Returns:
            Precio de equilibrio por kg (S/.).
        """
        return 13.2 / 1.2
    
    def cost_minimization(self, x: List[float]) -> float:
        """
        Modelo de minimizacion de costos para produccion de quinua.
        Ecuacion (6) del articulo.
        
        Args:
            x: Lista de 12 factores de costo [x1 a x12]
            
        Returns:
            Costo total minimizado (S/.).
            
        Raises:
            ValueError: Si el vector de entrada no tiene 12 elementos.
        """
        if len(x) != 12:
            raise ValueError("El vector de entrada debe tener 12 elementos")
            
        # Desempaquetar variables de costo
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = x
        
        # Calcular costo total
        return (x1 * x2 + 1237 * x3 + 70 * x4 + 5.8 * x5 + 45 * x6 + 
                75.38 * x7 + 21.67 * x8 + 325 * x9 + 3.65 * x10 + 
                11940 * x11 + 7822 * x12)
    
    def cost_minimization_constraints(self) -> dict:
        """
        Devuelve las restricciones para el problema de minimizacion de costos.
        Basado en la seccion III del articulo.
        
        Returns:
            Diccionario con limites de restriccion para cada variable.
        """
        return {
            'x1': {'max': 125},    # Maximo de jornales
            'x3': {'equal': 1.325},# Semilla (kg)
            'x4': {'min': 20},     # Fertilizante (kg)
            'x5': {'equal': 311},  # Abono (kg)
            'x6': {'equal': 1.5},  # Combustible (gal)
            'x7': {'equal': 3.25}, # Mantenimiento (horas)
            'x8': {'min': 20},     # Herbicida (ml)
            'x9': {'equal': 2},   # Renta de maquinaria (dias)
            'x10': {'min': 50},    # Servicios (horas)
            'x11': {'equal': 0.05},# Tasa de interes
            'x12': {'equal': 0.09} # Impuestos
        }


class GeneticAlgorithmOptimizer:
    """Optimizador con Algoritmo Genetico para produccion de quinua.
    
    Implementa:
    - Seleccion sexual o por torneo
    - Cruce aritmetico
    - Mutacion gaussiana
    - Elitismo
    
    Args:
        population_size: Tamano de la poblacion (default: 100)
        crossover_prob: Probabilidad de cruce (default: 0.65)
        mutation_prob: Probabilidad de mutacion (default: 0.08)
        selection_method: 'sexual' o 'tournament' (default: 'sexual')
    """
    
    def __init__(self, population_size: int = 100, crossover_prob: float = 0.65, 
                 mutation_prob: float = 0.08, selection_method: str = 'sexual'):
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.selection_method = selection_method
        self.model = QuinoaOptimizationModel()  # Modelo de quinua
        
    def fitness(self, individual: Tuple[float, float]) -> float:
        """
        Funcion de aptitud para el algoritmo genetico.
        Utilidad escalada: 1200*X1*X2 - 13200*X1
        
        Args:
            individual: Tupla (x1, x2) donde:
                x1: Terreno cultivado (hectareas)
                x2: Precio de venta (S/./kg)
                
        Returns:
            Valor de aptitud (utilidad escalada).
        """
        x1, x2 = individual
        return 1200 * x1 * x2 - 13200 * x1
    
    def satisfies_constraints(self, individual: Tuple[float, float], x3: float = 150) -> bool:
        """
        Verifica si un individuo satisface las restricciones del modelo.
        
        Args:
            individual: Tupla (x1, x2)
            x3: Demanda (default: 150 segun articulo)
            
        Returns:
            True si satisface todas las restricciones, False en caso contrario.
        """
        x1, x2 = individual
        x4 = 0.25  # Produccion de competidores fija
        
        # Restricciones de capacidad
        if x1 > 100 or x3 > 150:
            return False
            
        # Restriccion de precio maximo
        price_limit = 34 - (20.4 * x1) / x3 + 17 * x4
        if x2 > price_limit:
            return False
            
        return True
    
    def initialize_population(self) -> List[Tuple[float, float]]:
        """
        Inicializa una poblacion aleatoria dentro de los limites factibles.
        
        Returns:
            Lista de individuos (cada uno es una tupla x1, x2).
        """
        population = []
        for _ in range(self.population_size):
            x1 = random.uniform(0, 100)   # Hectareas (0 a maximo)
            x2 = random.uniform(10, 30)   # Precio (rango razonable)
            population.append((x1, x2))
        return population
    
    def sexual_selection(self, population: List[Tuple[float, float]], 
                         fitness_values: List[float]) -> List[Tuple[float, float]]:
        """
        Operador de seleccion sexual: empareja individuos por aptitud.
        
        Args:
            population: Poblacion actual
            fitness_values: Valores de aptitud de cada individuo
            
        Returns:
            Padres seleccionados para reproduccion.
        """
        # Ordenar poblacion por aptitud (descendente)
        sorted_pop = [x for _, x in sorted(zip(fitness_values, population), 
                      key=lambda pair: pair[0], reverse=True)]
        
        # Emparejar mejores individuos (1° con 2°, 3° con 4°, etc.)
        parents = []
        for i in range(0, len(sorted_pop) - 1, 2):
            parents.append((sorted_pop[i], sorted_pop[i+1]))
            
        return parents
    
    def tournament_selection(self, population: List[Tuple[float, float]], 
                             fitness_values: List[float], tournament_size: int = 3) -> List[Tuple[float, float]]:
        """
        Operador de seleccion por torneo: elige ganadores en torneos aleatorios.
        
        Args:
            population: Poblacion actual
            fitness_values: Valores de aptitud
            tournament_size: Tamano del torneo (default: 3)
            
        Returns:
            Padres seleccionados para reproduccion.
        """
        parents = []
        for _ in range(len(population) // 2):
            # Seleccionar participantes aleatorios
            tournament = random.sample(list(zip(population, fitness_values)), tournament_size)
            # Elegir los dos mejores
            tournament.sort(key=lambda x: x[1], reverse=True)
            parents.append((tournament[0][0], tournament[1][0]))
        return parents
    
    def crossover(self, parent1: Tuple[float, float], 
                 parent2: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Cruce aritmetico entre dos padres para producir dos hijos.
        
        Args:
            parent1: Primer padre
            parent2: Segundo padre
            
        Returns:
            Dos hijos (nuevos individuos).
        """
        if random.random() > self.crossover_prob:
            return parent1, parent2  # Sin cruce
            
        # Cruce aritmetico con factor aleatorio
        alpha = random.random()
        child1 = (alpha * parent1[0] + (1 - alpha) * parent2[0],
                 alpha * parent1[1] + (1 - alpha) * parent2[1])
        child2 = ((1 - alpha) * parent1[0] + alpha * parent2[0],
                 (1 - alpha) * parent1[1] + alpha * parent2[1])
        return child1, child2
    
    def mutate(self, individual: Tuple[float, float]) -> Tuple[float, float]:
        """
        Aplica mutacion con cambios aleatorios pequenos (distribucion gaussiana).
        
        Args:
            individual: Individuo a mutar
            
        Returns:
            Individuo mutado.
        """
        if random.random() > self.mutation_prob:
            return individual  # Sin mutacion
            
        x1, x2 = individual
        # Ruido gaussiano con limites de factibilidad
        x1 = max(0, min(100, x1 + random.gauss(0, 5)))
        x2 = max(10, min(30, x2 + random.gauss(0, 2)))
        return (x1, x2)
    
    def run(self, generations: int = 1000) -> Tuple[Tuple[float, float], float, List[float]]:
        """
        Ejecuta el algoritmo genetico de optimizacion.
        
        Args:
            generations: Numero de generaciones a ejecutar
            
        Returns:
            Tupla con:
                - Mejor individuo encontrado
                - Mejor aptitud (utilidad)
                - Historial de aptitudes por generacion
        """
        population = self.initialize_population()
        best_individual = None
        best_fitness = -float('inf')  # Inicializar con valor muy bajo
        fitness_history = []
        
        for gen in range(generations):
            # Evaluar aptitud (considerando restricciones)
            fitness_values = [
                self.fitness(ind) if self.satisfies_constraints(ind) else -float('inf') 
                for ind in population
            ]
            
            # Rastrear mejor individuo
            current_best = max(fitness_values)
            if current_best > best_fitness:
                best_idx = fitness_values.index(current_best)
                best_individual = population[best_idx]
                best_fitness = current_best
                
            fitness_history.append(best_fitness)
            
            # Seleccionar padres segun metodo configurado
            if self.selection_method == 'sexual':
                parents = self.sexual_selection(population, fitness_values)
            else:
                parents = self.tournament_selection(population, fitness_values)
                
            # Crear nueva generacion mediante cruce y mutacion
            new_population = []
            for parent1, parent2 in parents:
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(child1))
                new_population.append(self.mutate(child2))
                
            # Elitismo: preservar el mejor individuo
            if best_individual not in new_population:
                new_population[0] = best_individual
                
            population = new_population
            
        return best_individual, best_fitness, fitness_history


class QuinoaNeuralNetwork:
    """Red Neuronal para predecir utilidad en produccion de quinua.
    
    Arquitectura segun el articulo:
    - Capas densas con activacion tanh
    - Regularizacion L2
    - Dropout para prevenir sobreajuste
    
    Args:
        input_dim: Numero de caracteristicas de entrada (default: 4)
    """
    
    def __init__(self, input_dim: int = 4):
        self.model = Sequential([
            Dense(64, input_dim=input_dim, activation='tanh', 
                 kernel_regularizer=l2(0.01)),
            Dropout(0.1),
            Dense(64, activation='tanh', kernel_regularizer=l2(0.01)),
            Dense(1)  # Capa de salida lineal
        ])
        
        self.optimizer = Adam(learning_rate=0.0001)
        self.model.compile(optimizer=self.optimizer, loss='mse')  # Error Cuadratico Medio
        
    def generate_training_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera datos sinteticos de entrenamiento con distribucion normal.
        
        Args:
            n_samples: Numero de muestras a generar
            
        Returns:
            Tupla con:
                X: Caracteristicas (muestras x 4)
                y: Utilidad objetivo (muestras x 1)
        """
        X = np.zeros((n_samples, 4))
        # Generar caracteristicas con distribuciones normales
        X[:, 0] = np.random.normal(50, 20, n_samples)   # x1: hectareas
        X[:, 1] = np.random.normal(17, 5, n_samples)    # x2: precio de venta
        X[:, 2] = np.random.normal(100, 30, n_samples)  # x3: demanda
        X[:, 3] = np.random.normal(0.25, 0.05, n_samples)  # x4: competencia
        
        # Aplicar limites de factibilidad
        X[:, 0] = np.clip(X[:, 0], 0, 100)
        X[:, 1] = np.clip(X[:, 1], 10, 30)
        X[:, 2] = np.clip(X[:, 2], 0, 150)
        X[:, 3] = np.clip(X[:, 3], 0, 0.5)
        
        # Calcular utilidad objetivo (Eq. 4)
        y = 1.2 * X[:, 0] * X[:, 1] - 13.2 * X[:, 0]
        
        return X, y.reshape(-1, 1)
    
    def train(self, X: np.ndarray, y: np.ndarray, 
             epochs: int = 200, batch_size: int = 10) -> tf.keras.callbacks.History:
        """
        Entrena el modelo de red neuronal.
        
        Args:
            X: Caracteristicas de entrenamiento
            y: Valores objetivo
            epochs: Epocas de entrenamiento
            batch_size: Tamano del lote
            
        Returns:
            Historial de entrenamiento.
        """
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            X: Caracteristicas de entrada
            
        Returns:
            Predicciones del modelo.
        """
        return self.model.predict(X)


def main():
    """Funcion principal que ejecuta todos los modelos de optimizacion."""
    print("Modelo de Optimizacion de Produccion de Quinua para la Region de Puno")
    print("Basado en investigacion de la Universidad Nacional del Altiplano, Puno")
    
    # 1. Maximizacion de utilidad y analisis de punto de equilibrio
    print("\n=== Maximizacion de Utilidad y Analisis de Punto de Equilibrio ===")
    model = QuinoaOptimizationModel()
    
    # Solucion optima del articulo
    optimal_hectares = 100
    optimal_price = 22.10
    max_utility = model.utility_maximization(optimal_hectares, optimal_price)
    print(f"Solucion optima del articulo: {optimal_hectares} ha a S/. {optimal_price}/kg")
    print(f"Utilidad maxima: S/. {max_utility * 1000:,.2f} (escalado)")
    
    # Precio de punto de equilibrio
    break_even_price = model.calculate_break_even_price()
    print(f"\nPrecio de punto de equilibrio: S/. {break_even_price:.2f}/kg")
    
    # 2. Optimizacion con algoritmo genetico
    print("\n=== Optimizacion con Algoritmo Genetico ===")
    print("Ejecutando algoritmo genetico con seleccion sexual...")
    ga_sexual = GeneticAlgorithmOptimizer(selection_method='sexual')
    best_ind, best_fit, fitness_history = ga_sexual.run(generations=100)
    
    print(f"Mejor solucion encontrada: {best_ind[0]:.2f} ha a S/. {best_ind[1]:.2f}/kg")
    print(f"Mejor utilidad: S/. {best_fit * 1000:,.2f}")
    
    # 3. Enfoque con red neuronal
    print("\n=== Enfoque con Red Neuronal ===")
    nn_model = QuinoaNeuralNetwork()
    X_train, y_train = nn_model.generate_training_data()
    print("Entrenando red neuronal...")
    history = nn_model.train(X_train, y_train, epochs=200, batch_size=10)
    
    # Prediccion de prueba
    test_input = np.array([[optimal_hectares, optimal_price, 150, 0.25]])
    prediction = nn_model.predict(test_input)
    print(f"\nUtilidad predicha para solucion optima: S/. {prediction[0][0] * 1000:,.2f}")
    
    # 4. Minimizacion de costos (ejemplo simplificado)
    print("\n=== Minimizacion de Costos ===")
    # Valores de las restricciones del articulo
    cost_vars = [
        125,    # x1: Jornales
        41.2,   # x2: Salario diario (S/.)
        1.325,  # x3: Semilla (kg)
        20,     # x4: Fertilizante (kg)
        311,    # x5: Abono (kg)
        1.5,    # x6: Combustible (gal)
        3.25,   # x7: Mantenimiento (horas)
        20,     # x8: Herbicida (ml)
        2,      # x9: Renta de maquinaria (dias)
        50,     # x10: Servicios (horas)
        0.05,   # x11: Tasa de interes
        0.09    # x12: Impuestos
    ]
    min_cost = model.cost_minimization(cost_vars)
    print(f"Costo minimo de produccion por hectarea: S/. {min_cost:,.2f}")


if __name__ == "__main__":
    main()