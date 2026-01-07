# Judge's Familiar – Introducción del Proyecto

**Judge's Familiar** es un asistente de IA diseñado para actuar como el compañero definitivo de reglas para jugadores de *Magic: The Gathering*. Más que un simple buscador, funciona como un **experto consultor en tiempo real**, capaz de interpretar dudas en lenguaje natural y explicar interacciones complejas basándose estrictamente en la documentación oficial.

El núcleo del sistema utiliza una arquitectura **RAG (Retrieval-Augmented Generation)** que combina:

* Las **Comprehensive Rules (CR)** oficiales de Magic (incluyendo el Glosario).
* El **Oracle text** de la base de datos de cartas (vía MTGJSON).

A diferencia de los LLMs genéricos que pueden "alucinar" reglas, *Judge's Familiar* recupera las normas exactas y construye una respuesta razonada, garantizando **trazabilidad y precisión**.

## Objetivo de la Versión 1 (V1)

La primera versión se define como un **"Pocket Companion"** (Compañero de Bolsillo) enfocado en la resolución de dudas técnicas.

* **Entrada Multimodal:**
    * Texto (consultas directas).
    * Voz (transcripción automática mediante modelo **Whisper**).
* **Salida Transparente:**
    * Explicación pedagógica de la interacción ("*Por qué* ocurre esto").
    * **Citas explícitas obligatorias** (ej. `[702.19b]`, `[510.1c]`).
    * Referencias cruzadas entre definiciones del Glosario y reglas numéricas.
* **Filosofía de Diseño:**
    * **Objetividad:** El sistema explica la mecánica, no juzga la conducta de los jugadores.
    * **Rigor:** Si la información no existe en las reglas recuperadas, el sistema lo indica en lugar de inventar.

*Judge's Familiar* no pretende reemplazar al juez humano en política de torneos o disputas de conducta; su misión es **democratizar el acceso a las reglas**, permitiendo partidas más fluidas y justas tanto en entornos casuales como competitivos.

## Arquitectura Técnica

El sistema se estructura en un pipeline de cuatro capas:

1.  **Ingesta de Datos (ETL)**
    * Parsing atómico de las *Comprehensive Rules* y el *Glosario* (1 regla = 1 nodo).
    * Indexación de cartas y textos Oracle.
2.  **Índices Semánticos**
    * Base de datos vectorial optimizada para búsquedas de similitud (embeddings).
3.  **Motor RAG + LLM**
    * Recuperación híbrida de reglas y definiciones.
    * Generación de respuesta con *Prompt Engineering* estricto (rol de Asistente Nivel 3).
    * Uso de modelos eficientes (`gpt-4o-mini`) con temperatura 0.0 para máxima fidelidad.
4.  **Interfaz de Usuario**
    * Chat web responsive (Desktop/Móvil).
    * Visualización clara de la respuesta separada de las fuentes técnicas.

## Escalabilidad (Roadmap)

La arquitectura modular permite futuras extensiones sin reescribir el núcleo:

* **Reconocimiento Visual:** Identificación de cartas físicas mediante cámara (OCR/Image Recognition).
* **Búsqueda Semántica por Arte:** Localización de cartas basada en descripciones visuales (*"bestia verde con tres cabezas"*).
* **Agentes de Decisión:** Una capa superior opcional para simular juicios complejos encadenados.

## Experiencia Interactiva

Como parte de la presentación, se entregarán copias físicas de la carta *Judge's Familiar* modificadas con un **código QR dinámico**. Esto permitirá a la audiencia y al jurado escanear la carta y **probar el sistema en vivo desde sus propios dispositivos**, cerrando la brecha entre el juego físico (Tabletop) y la asistencia digital.
