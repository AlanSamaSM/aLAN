# aLAN
Your everyday companion

process: Prototipo Paso a Paso: Clonar tu Voz con Sesame CSM e Integrarla con un Modelo de Lenguaje
Introducción
El Conversational Speech Model (CSM) de Sesame es un modelo de generación de voz diseñado para lograr conversaciones más naturales con inteligencia artificial
digitalocean.com
. A diferencia de voces sintéticas rígidas, CSM busca incorporar matices humanos (pausas, entonación, tono emocional, modismos, etc.) en el habla generada, para que sientas que hablas con alguien y no con algo
digitalocean.com
digitalocean.com
. En este prototipo, clonaremos tu propia voz usando CSM y la usaremos para dar vida a las respuestas generadas por un modelo de lenguaje (LLM). Todo el proceso será local, aprovechando hardware personal (GPU) y evitando servicios en la nube en la medida de lo posible. ¿Qué lograremos? Al final de estos pasos podrás ingresar una pregunta (por texto o voz) a un asistente de IA local; este la responderá con texto mediante un LLM, y luego CSM convertirá esa respuesta en audio con tu voz clonada de forma expresiva. Esto te permitirá tener una conversación más natural con la IA, escuchando respuestas habladas en un tono familiar.
Requisitos Previos
Antes de comenzar, asegúrate de contar con lo siguiente:
Hardware: Una computadora con GPU compatible con CUDA (NVIDIA) y al menos ~8 GB de VRAM para ejecutar el modelo eficientemente
reddit.com
reddit.com
. (CSM puede correr en CPU o Apple Silicon MPS, pero mucho más lento).
Sistema y Lenguaje: Python 3.10 (recomendado) en un entorno Linux/macOS. Nota: En Windows es posible usarlo, pero requiere ajustes menores (ver más abajo)
github.com
github.com
.
Librerías y dependencias: PyTorch 2.4+ con soporte CUDA, Torchaudio 2.4, Transformers 4.49+, Hugging Face Hub, etc. (todas incluidas en los requerimientos del proyecto)
github.com
. También podría requerirse tener ffmpeg instalado para ciertas operaciones de audio
github.com
.
Modelos de Hugging Face: Acceso a los pesos del modelo de voz Sesame CSM-1B y al backbone Llama-3.2-1B en Hugging Face
github.com
. Estos repositorios son public gated, por lo que necesitas iniciar sesión en Hugging Face y aceptar los términos de uso antes de descargarlos
huggingface.co
github.com
. Crea un token de acceso (Read) de Hugging Face y tenlo a mano para autenticación.
Muestra de tu voz: Un clip de audio claro con tu voz, de preferencia 2-3 minutos de duración para mejor calidad de clonación
github.com
, con su transcripción exacta. Puedes grabar varias frases donde uses tu tono natural y modismos. Un micrófono de buena calidad ayudará a capturar detalles de tu voz.
Nota: El modelo CSM actualmente está centrado en inglés (solo soporta oficialmente un idioma en su versión base)
digitalocean.com
. Por contaminación de datos podría manejar algo de español, pero es posible que la calidad/prosodia en español no sea perfecta
github.com
. Aun así, tu muestra de voz en español puede ayudar a guiar el tono. Ten esto en cuenta al evaluar los resultados.
1. Instalación de Sesame CSM en tu Entorno Local
Primero, descargaremos e instalaremos el proyecto CSM de Sesame:
Clona el repositorio de GitHub y crea un entorno virtual:
bash
Copiar
Editar
git clone https://github.com/SesameAILabs/csm.git
cd csm
python3.10 -m venv .venv
source .venv/bin/activate
Luego instala las dependencias con pip:
bash
Copiar
Editar
pip install -r requirements.txt
Esto instalará PyTorch, torchaudio, transformers y otras librerías necesarias en las versiones probadas.
Configura variables de entorno necesarias: CSM utiliza un tokenizer de audio llamado Mimi que por defecto hace compilación just-in-time. Para evitar problemas, desactiva la compilación diferida estableciendo:
bash
Copiar
Editar
export NO_TORCH_COMPILE=1
(En Windows, en lugar de variable de entorno, puedes definir os.environ["NO_TORCH_COMPILE"]="1" al inicio de tu script Python, como ya se hace en el código de ejemplo.)
Inicio de sesión en Hugging Face: Autentícate para descargar los modelos requeridos. Ejecuta en terminal:
bash
Copiar
Editar
huggingface-cli login
Ingresa tu token de acceso de Hugging Face cuando lo pida. Esto permitirá que el código descargue los pesos de CSM-1B y Llama-3.2-1B automáticamente al usarlos
github.com
. Asegúrate de haber aceptado previamente los términos de ambos modelos en la web de Hugging Face, ya que son repositorios con acceso restringido que requieren tu confirmación
huggingface.co
github.com
.
(Opcional) Ajustes para Windows: Si usas Windows, es posible que la dependencia triton (usada para optimizaciones) no esté disponible. La guía oficial indica instalar una versión especial:
bash
Copiar
Editar
pip install triton-windows
Además, podrías necesitar editar el archivo requirements.txt o el script para omitir mlx si solo aplica a macOS (Metal Performance Shaders)
reddit.com
. En general, se han reportado casos de éxito en Windows 11 después de estos ajustes
reddit.com
reddit.com
, pero la ruta más fácil en Windows es usar WSL2 o Docker con un entorno Linux si surgen complicaciones.
Con esto, el entorno de CSM debería estar listo. Procederemos a realizar una primera prueba básica para asegurar que todo funcione antes de integrar tu voz.
2. Prueba Rápida de Conversión de Texto a Voz con CSM
Para verificar la instalación, ejecuta el script de ejemplo incluido:
bash
Copiar
Editar
python run_csm.py
Este script genera una breve conversación de prueba entre dos voces sintéticas incluidas como ejemplo. Internamente, carga el modelo CSM-1B y prepara dos prompts de voz (speaker A y speaker B) con muestras de audio provistas por Sesame
github.com
. Luego crea una lista de turnos de conversación de texto (alternando entre speaker 0 y 1) y los convierte a audio iterativamente. Al finalizar, obtendrás un archivo WAV de salida (por defecto full_conversation.wav) con todas las frases concatenadas
digitalocean.com
. Escucha ese archivo de salida para confirmar que el modelo funciona correctamente en tu sistema. Deberías oír dos voces distintas conversando (en inglés, con entonación natural y pausas). Si obtuviste este resultado, ¡felicidades! CSM está generando voz localmente. En caso de error (por ejemplo, falta de VRAM, problemas de dependencias), revísalos antes de continuar.
Tip: Si deseas personalizar la conversación de ejemplo, puedes editar las frases de la lista conversation en run_csm.py (líneas ~87-90) antes de ejecutar el script
digitalocean.com
. Esto te permite ver cómo CSM maneja distintos textos.
3. Preparación de una Muestra de Voz Personal
Ahora entraremos en materia de clonación de voz. Para que CSM imite tu forma de hablar, debemos proporcionarle una muestra de tu voz actuando como prompt o contexto de entrada. Sigue estos pasos:
Graba tu voz: Prepara un archivo de audio (.wav o .mp3) con tu voz hablando de forma natural. Lo ideal son varios párrafos (2-3 minutos) continuos
github.com
, para captar bien tus patrones de habla, entonación y timbre. Habla como lo harías en conversación, incluyendo tus muletillas o modismos típicos. Un ejemplo podría ser contar una anécdota breve o leer en voz alta un texto que incluya preguntas y exclamaciones (para captar entonaciones variadas).
Formato del audio: Asegúrate de que la grabación tenga buena calidad (poco ruido de fondo). CSM opera internamente a 24 kHz mono; si tu archivo está en otra tasa de muestreo, el código lo re-muestrea automáticamente al cargarlo
github.com
. Guarda el archivo como, por ejemplo, mi_voz.wav en el directorio del proyecto o anota su ruta completa.
Transcribe el contenido: Es crucial obtener el texto exacto de lo que dices en el audio. El modelo necesita el par audio+texto transcrito para entender cómo suenas al decir determinadas palabras. Transcribe manualmente lo que dijiste o usa una herramienta como Whisper si prefieres (asegurándote de corregir cualquier error para que coincida perfectamente)
github.com
. Guarda esa transcripción como una cadena de texto, por ejemplo:
text
Copiar
Editar
context_text = "Hola, soy [Tu Nombre]. Esta es una muestra de mi voz para probar el modelo CSM. Me gusta la tecnología y la inteligencia artificial... (etc.)"
Debe ser literalmente lo que está en el audio mi_voz.wav. Incluye puntuación adecuada, ya que pausas y entonaciones podrían depender de signos como comas, puntos o interrogaciones.
4. Clonación de tu Voz con CSM
Con tu audio de contexto listo, modificaremos el código para incorporarlo y generar voz sintética con tu timbre. No necesitas re-entrenar nada; CSM, al ser un modelo generativo base sin voces predefinidas, puede adaptarse a tu voz a partir de un ejemplo (few-shot voice cloning). Pasos para usar tu muestra en el modelo:
Carga del modelo: En un script nuevo (o modificando run_csm.py), primero carga el generador de CSM igual que en el ejemplo:
python
Copiar
Editar
import torch
from generator import load_csm_1b, Segment
import torchaudio

device = "cuda" if torch.cuda.is_available() else "cpu"
generator = load_csm_1b(device=device)
Esto prepara el modelo en la GPU (u CPU si no hay GPU). La instancia generator tendrá un método generate para convertir texto en audio.
Carga de tu audio y creación del Segment: Utiliza torchaudio para cargar tu archivo de voz y ajustarlo a 24 kHz. Luego crea un objeto Segment que combina tu audio con su transcripción, asignándole un ID de locutor (speaker). Por ejemplo, podríamos usar speaker=0 para representarte. Ejemplo de código:
python
Copiar
Editar
# Cargar audio de muestra de voz
audio_waveform, sr = torchaudio.load("mi_voz.wav")
audio_waveform = audio_waveform.squeeze(0)
# Remuestrear si es necesario a la tasa del modelo (24 kHz)
audio_waveform = torchaudio.functional.resample(audio_waveform, orig_freq=sr, new_freq=generator.sample_rate)
# Crear el segmento de contexto con tu voz
prompt_myvoice = Segment(text=context_text, speaker=0, audio=audio_waveform)
Aquí context_text es la transcripción exacta preparada en el paso anterior, y audio_waveform es tu audio como tensor. Ahora prompt_myvoice representa tu voz hablando ese texto específico.
Generar voz clonada a partir de texto: Ya con el segmento de tu voz como contexto, usemos CSM para generar una línea de diálogo nueva imitando ese estilo. Podemos probar con una frase sencilla, por ejemplo: "Hola, esta es una prueba de mi voz clonada." Para ello, llamamos a generator.generate proporcionando el texto, el ID de speaker 0 (el que corresponde a tu voz) y el contexto:
python
Copiar
Editar
texto_respuesta = "Hola, esta es una prueba de mi voz clonada."
audio_generado = generator.generate(
    text=texto_respuesta,
    speaker=0,
    context=[prompt_myvoice],       # contexto con tu voz
    max_audio_length_ms=10000      # duración máxima de 10s, ajustar según frase
)
torchaudio.save("respuesta.wav", audio_generado.unsqueeze(0).cpu(), generator.sample_rate)
El resultado será un archivo respuesta.wav con el audio sintético. Reprodúcelo: deberías escuchar tu voz diciendo la frase de prueba. Idealmente, notarás que el timbre y ritmo son similares a tu habla original. CSM aprovecha el contexto dado para ajustar la salida; de hecho, funciona mejor con contexto que sin él
github.com
. Al proporcionarle tu muestra de voz como segmento de contexto, orientamos al modelo a generar con ese timbre y estilo.
Referencia: El uso de Segment y contexto sigue el patrón de los ejemplos oficiales. CSM genera audio de mayor calidad cuando se le da contexto adecuado de conversación o voz base
github.com
. En el ejemplo, pasamos [prompt_myvoice] en el parámetro context para guiar la síntesis con tu voz.
Verifica la calidad: Escucha respuesta.wav. La voz debería ser reconociblemente tuya, aunque tal vez no idéntica al 100%. Según reportes, CSM captura muchas características de la voz y el resultado es decente pero no perfecto en cuanto a clonación
github.com
. Si la entonación suena un poco plana, ten en cuenta que CSM aún es un modelo base sin fine-tuning específico; a pesar de ello, suele mantener el timbre y algunos rasgos particulares de la voz de contexto.
(Opcional) Ajusta y refina: Si no estás satisfecho con la similitud, podrías intentar:
Proveer un contexto más largo: incluye más segmentos de tu voz. Por ejemplo, podrías dividir tu audio en 2-3 segmentos separados con sus textos (todos con el mismo speaker ID) y pasarlos todos en la lista de context. Más datos de referencia podrían ayudar al modelo a afinar la consistencia de voz y expresividad.
Asegurarte de que la transcripción tiene puntuación adecuada: Las comas, pausas y signos de exclamación o interrogación en el texto de contexto podrían enseñar mejor tus inflexiones al modelo.
Probar con diferentes frases para ver cómo maneja entonaciones variadas (de alegría, pregunta, énfasis). CSM intenta reflejar contexto conversacional, por lo que cuanta más variedad haya en tu muestra, más rico puede ser el resultado.
5. Integración con un Modelo de Lenguaje (LLM) para Respuestas Dinámicas
Hasta ahora hemos generado manualmente una frase con tu voz clonada. El siguiente paso es automatizar las respuestas usando un Language Model que genere el texto de la respuesta, para luego pasarlo a CSM. Recordemos que CSM no genera texto por sí mismo ni responde preguntas; solo convierte texto a habla. Para el cerebro de la conversación utilizaremos un LLM separado
github.com
. ¿Qué LLM usar? Dado que deseamos una solución local sin depender de API externas, puedes optar por un modelo abierto como Llama 2 (Meta AI), GPT4All, Vicuna, etc., según tu hardware. Un modelo de ~7B o 13B parámetros afinado para chat podría correr en una GPU de gama alta o incluso CPU (con menos rapidez). La integración se puede hacer de varias formas:
Usando la biblioteca Transformers de Hugging Face para cargar un modelo pre-entrenado en modo texto (por ejemplo, AutoModelForCausalLM con un tokenizer correspondiente). Luego generar respuestas con model.generate() dado el prompt de conversación.
Usando una interfaz existente tipo LangChain o LLM WebUI para obtener respuestas de chat.
O incluso mediante API locales como llama.cpp, dependiendo de tu preferencia, siempre que puedas obtener el texto de respuesta de forma programática.
Para ilustrar el flujo, asumamos que tenemos una función generar_respuesta_LLM(prompt_texto) que recibe la pregunta del usuario en texto y devuelve una respuesta en texto usando un LLM local. La integración con CSM sería entonces:
Entrada del usuario: El usuario plantea una pregunta o mensaje. Podría ser vía texto escrito o mediante voz. Si es por voz real, necesitaríamos un módulo de speech-to-text (por ejemplo, Whisper local) para transcribirla a texto primero. Para simplificar el prototipo inicial, consideremos que la entrada del usuario es texto escrito en la consola o en una interfaz.
Generación de respuesta con LLM: Pasamos el mensaje del usuario al LLM. Ejemplo pseudo-código:
python
Copiar
Editar
pregunta = "¿Cómo estará el clima hoy en la tarde?"
respuesta_texto = generar_respuesta_LLM(pregunta)
print("LLM:", respuesta_texto)
Supongamos que respuesta_texto devuelve: "El clima estará soleado y cálido, con aproximadamente 25 grados centígrados." (en función del modelo y su conocimiento). Este texto es la respuesta que queremos que la IA pronuncie con voz clonada.
Síntesis de voz con CSM (tu voz): Ahora llamamos a CSM para convertir respuesta_texto en audio, usando nuevamente tu voz como speaker. Dado que ya preparamos el segmento prompt_myvoice antes (que contiene tu voz de referencia), podemos reutilizarlo en el contexto para mantener la identidad de voz. Por ejemplo:
python
Copiar
Editar
audio_respuesta = generator.generate(
    text=respuesta_texto,
    speaker=0,                # 0 corresponde a tu voz clonada
    context=[prompt_myvoice], # contexto base con tu voz
    max_audio_length_ms=20000 # ajustar si la respuesta es larga
)
torchaudio.save("respuesta_llm.wav", audio_respuesta.unsqueeze(0).cpu(), generator.sample_rate)
Aquí generamos hasta 20 segundos de audio por seguridad. Puedes también dividir respuestas muy extensas en varios trozos si fuera necesario.
Contexto conversacional: En una interacción multi-turno (varias idas y vueltas), es recomendable incluir todo el historial relevante en el contexto al generar cada respuesta, no solo el prompt inicial. CSM permite encadenar segmentos previos para lograr continuidad en la conversación
github.com
. Por ejemplo, podrías mantener una lista historial_segmentos donde agregas:
Segmento de la última pregunta del usuario (texto del usuario, potencialmente con audio si estuviera disponible, aunque sea TTS del usuario o simplemente omitido si no hay audio).
Segmento de la última respuesta de la IA (texto + audio generado previo).
Luego, al generar la siguiente respuesta, usar context = [prompt_myvoice] + historial_segmentos. Esto podría ayudar a que la voz de la IA conserve consistencia e incluso adaptar ligeramente la entonación según el flujo (por ejemplo, respondiendo con tono amigable si la conversación previa así lo sugiere). CSM “considera el historial de la conversación” para producir habla más contextual y natural
digitalocean.com
, aunque recuerda que el contenido de la respuesta lo decide el LLM.
Reproducir/Entregar el audio: El archivo respuesta_llm.wav contendrá la respuesta hablada por la IA con tu voz clonada. Puedes reproducirlo automáticamente en tu aplicación o interfaz, o enviarlo a algún sistema de chat de voz. Dado que todo se ejecutó localmente, la latencia dependerá principalmente de la velocidad de tu LLM y de CSM generando audio. CSM no opera en tiempo real (puede tardar unos segundos en sintetizar dependiendo de la longitud del audio y la GPU), pero el retraso aún es manejable para un intercambio conversacional pausado.
Ideas de Integración Adicionales
Interfaz de usuario: Para mejorar la usabilidad, podrías envolver este flujo en una interfaz gráfica simple. Por ejemplo, usar Gradio o una aplicación web ligera donde el usuario hable o escriba, y la respuesta de audio se reproduzca automáticamente. De hecho, hay proyectos comunitarios que ya han creado frontends para CSM (por ejemplo, un demo con Gradio que corre localmente)
reddit.com
reddit.com
. Estas interfaces permiten grabar tu voz, enviar la pregunta al LLM y escuchar la respuesta de forma más interactiva.
Entrada por voz: Integrar un paso de reconocimiento de voz local (como Whisper en modo offline) para que el usuario no tenga ni que escribir. El pipeline completo sería: voz del usuario (micrófono) → transcripción (texto) → LLM genera texto respuesta → CSM genera voz respuesta → audio de salida. Todo puede montarse localmente con las librerías adecuadas.
Múltiples voces o personalidades: CSM soporta múltiples speakers. Podrías clonar no solo tu voz sino también otras (pareja, amigo, voz de celebridad con permiso, etc.) y asignar cada una a un ID distinto. Luego, con un solo modelo, generar diálogos donde la IA adopte distintos personajes. Por ejemplo, speaker 0 = tu voz, speaker 1 = otra voz; el LLM podría generar diálogos completos y CSM los vocifera asignando cada turno al speaker correspondiente.
Ajustes de calidad: Ten en cuenta que este es un modelo de investigación. No viene pre-entrenado en voces específicas
github.com
, lo que es una ventaja en flexibilidad pero también implica que la clonación se basa solo en tu prompt. Si en el futuro Sesame libera versiones fine-tuned (como mencionaron con voces llamadas Maya y Miles para demo expresivo
digitalocean.com
), podrías aprovechar esas mejoras. Asimismo, están en planes de soportar muchos más idiomas, lo que beneficiará las conversaciones en español
digitalocean.com
. Mantente atento a actualizaciones del modelo.
6. Prueba del Prototipo y Conclusiones
Llegó el momento de probar el sistema completo. Inicia una conversación con tu asistente IA local y observa los resultados:
Haz una pregunta o saludo al asistente (ej. "Hola, ¿cómo estás?").
El LLM debería generar una respuesta coherente en texto, por ejemplo: "¡Hola! Me siento muy bien hoy, gracias por preguntar." (La calidad dependerá del modelo lingüístico elegido).
CSM toma esa respuesta y la sintetiza en audio con tu voz clonada. Escucharás algo como tu propia voz diciendo esa frase, posiblemente con entonación amigable si el contexto se mantiene.
Continúa la interacción con más preguntas para ver cómo maneja el hilo de conversación. Observa si la voz mantiene consistencia y si hay variaciones naturales en la entonación conforme avanzan los turnos. Este es precisamente el objetivo de CSM: “mejorar la conversación digital incorporando pausas cálidas, cambios sutiles de tono y respuestas pensativas que apreciamos en la conversación humana”
digitalocean.com
.
En términos de resultados, no esperes una clonación perfecta al primer intento. Te darás cuenta de que, aunque la voz suena como la tuya, puede carecer de algo de naturalidad o tener ligeras distorsiones en pronunciación de ciertos nombres o palabras inusuales (especialmente en español) debido a las limitaciones lingüísticas actuales. Aún así, es sorprendente oír una respuesta generada por IA con un timbre familiar. Con ajustes y futuras mejoras del modelo (o combinándolo con técnicas de fine-tuning), el realismo podrá incrementarse. Consideraciones finales: Este prototipo es completamente local, lo que significa que la privacidad de tu voz y conversaciones se mantiene — un aspecto importante dado que la clonación de voz puede ser delicada. Recuerda usar esta tecnología de forma ética: el propio proyecto CSM advierte contra usos maliciosos como suplantación de identidad o desinformación
github.com
. Siempre obtén consentimiento si planeas clonar la voz de alguien más, y sé transparente en aplicaciones que usen voces sintéticas. ¡Disfruta experimentando con tu nuevo asistente con voz personalizada! Este enfoque abre la puerta a interacciones hombre-máquina mucho más humanizadas, y aunque aún no es perfecto, demuestra el potencial de combinar modelos de lenguaje con modelos de voz avanzados. Fuentes Utilizadas:
Documentación oficial de CSM (Sesame AI Labs)
github.com
github.com
github.com
Artículo técnico de DigitalOcean sobre Sesame CSM
digitalocean.com
digitalocean.com
digitalocean.com
Repositorio CSM Voice Cloning (isaiahbjork) para consejos de clonación
github.com
github.com
Hilo informativo en Reddit sobre ejecución local de CSM
reddit.com
reddit.com
.