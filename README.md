# Progetto di Elasticità Non Lineare

Progetto di studio e simulazione dell'elasticità non lineare con particolare focus sui modelli costitutivi di Mooney-Rivlin e Gent.

## Contenuto

- **Modelli Analitici**: Implementazione Python dei modelli di Mooney-Rivlin e Gent
- **Simulazioni ANSYS**: File APDL per simulazioni agli elementi finiti
- **Analisi Parametriche**: Script per lo sweep di parametri e pressioni
- **Confronti**: Visualizzazione comparativa tra modelli diversi

## File Principali

- `analytical_mooney_rivlin.py` - Modello analitico Mooney-Rivlin
- `analytical_gent.py` - Modello analitico Gent
- `cilinder_mooney_rivlin.apdl` - Simulazione cilindro Mooney-Rivlin (ANSYS)
- `cilinder_gent.apdl` - Simulazione cilindro Gent (ANSYS)
- `pressure_sweep_mooney_rivlin.py` - Analisi con variazione di pressione (Mooney-Rivlin)
- `pressure_sweep_gent.py` - Analisi con variazione di pressione (Gent)
- `confronto_modelli.py` - Confronto tra i modelli

## Risultati

Grafici e risultati delle simulazioni:
- `confronto_modelli.jpeg` - Confronto visivo dei modelli
- `grafici_mooney_rivlin.jpeg` - Risultati Mooney-Rivlin
- `sweep_pressure_*.jpeg` - Analisi parametriche
