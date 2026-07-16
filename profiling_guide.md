# Guía de perfilado GPU: entendiendo qué hace el kernel por dentro

Este documento explica el proceso de perfilado del kernel CUDA para el método de Jacobi unilateral. Está escrito asumiendo que sabes programar en CUDA pero no necesariamente sabes cómo funciona el hardware por dentro. La idea es que entiendas **qué mide cada métrica**, **por qué da esos valores** y **qué implicancias tiene** para el rendimiento.

---

## 1. ¿Qué es perfilar y por qué hacerlo?

Perfilar es medir qué hace tu programa **mientras se ejecuta en la GPU**, usando herramientas que leen contadores físicos dentro del chip. Es la diferencia entre decir "el kernel tarda 16 ms" y entender **por qué** tarda 16 ms: ¿está esperando datos de la memoria? ¿están todos los núcleos ocupados? ¿hay cuellos de botella ocultos?

Sin perfilado, optimizar es adivinar. Con perfilado, sabes exactamente dónde está el límite.

---

## 2. Las herramientas: ncu y nsys

NVIDIA provee dos perfiladores complementarios:

| Herramienta | Qué mide | Cuándo usarla |
|-------------|----------|---------------|
| **Nsight Systems** (`nsys`) | Línea de tiempo completa: kernels, transferencias CPU↔GPU, sincronización | Para ver el panorama general |
| **Nsight Compute** (`ncu`) | Contadores de hardware dentro de un solo kernel: ocupación, ancho de banda, cachés | Para optimizar un kernel específico |

En este proyecto usamos `ncu` porque ya sabíamos que el kernel domina el tiempo total (el test de estrés mostró que las transferencias son <1% del tiempo para datasets grandes). La pregunta era: **dentro del kernel, ¿qué está pasando?**

### Cómo se ejecuta ncu

```bash
ncu --set full ./svd_cuda data/BetaPic/center_im.fits
```

`ncu` re-ejecuta el kernel múltiples veces, cada vez midiendo un conjunto distinto de contadores de hardware. Al final entrega un reporte consolidado. Es lento (puede tomar varios minutos) pero extremadamente detallado.

---

## 3. Arquitectura mínima de GPU que necesitas entender

Para leer un perfil de `ncu` hay que saber cuatro conceptos:

### 3.1. SMs (Streaming Multiprocessors)

La GPU está dividida en **SMs**. Piensa en ellos como mini-CPUs independientes. Tu RTX 3060 Laptop tiene **30 SMs**. Cada SM ejecuta bloques de hilos. Si lanzas menos bloques que SMs, algunos SMs se quedan sin trabajo.

Nuestro kernel lanza `D/2` bloques. Para `median_unsat` (D=6), son 3 bloques: 27 de 30 SMs no hacen nada. Para `center_im` (D=192), son 96 bloques: cada SM recibe varios bloques y todos trabajan.

### 3.2. Warps

Dentro de un SM, los hilos no se ejecutan de a uno. Se agrupan en **warps** de 32 hilos que ejecutan la misma instrucción al mismo tiempo (modelo SIMT: *Single Instruction, Multiple Threads*). Nuestro kernel usa 256 hilos por bloque = 8 warps por bloque.

Un SM puede manejar hasta 48 warps simultáneamente (en RTX 3060). Con 6 bloques por SM (el límite que imponen los registros, como veremos), tenemos 6 × 8 = 48 warps. Eso da **ocupación teórica del 100%**: en teoría, todos los slots de warps están llenos.

### 3.3. Jerarquía de memoria

La GPU tiene varios tipos de memoria, ordenados de más rápida a más lenta:

| Tipo | Tamaño típico | Latencia | ¿Quién la usa? |
|------|--------------|----------|-----------------|
| **Registros** | 64K por SM | ~0 ciclos | Un hilo individual |
| **Shared memory** | 16-48 KB por SM | ~20 ciclos | Todos los hilos de un bloque |
| **Caché L1** | 128 KB por SM | ~30-100 ciclos | Automática |
| **Caché L2** | 3 MB compartida | ~200 ciclos | Automática |
| **DRAM (VRAM)** | 6 GB | ~400-800 ciclos | Toda la GPU |

La clave: **la DRAM es enormemente más lenta que todo lo demás**. Si tu kernel pasa la mayor parte del tiempo esperando datos de DRAM, estás *memory-bound*. Si pasa la mayor parte haciendo cuentas, estás *compute-bound*.

### 3.4. Acceso coalescido

Cuando 32 hilos de un warp leen 32 direcciones consecutivas de memoria, el hardware lo sirve en **una sola transacción**. Si leen direcciones dispersas, necesita hasta 32 transacciones separadas. Nuestro kernel usa layout *column-major*: el hilo 0 lee `X[0]`, el hilo 1 lee `X[1]`, etc. Son direcciones consecutivas = acceso coalescido perfecto. Esto es lo que permite alcanzar el 89% del ancho de banda pico.

---

## 4. El perfil del dataset pequeño (median_unsat, D=6)

### Configuración del kernel

```
Lanzamiento: <<<3 bloques, 256 hilos>>>
Matriz:      4096 filas × 6 columnas (después de transponer)
Datos leídos: 3 pares × 2 columnas × 4096 filas × 8 bytes = 192 KB
```

### Resultados y qué significan

#### Throughput de cómputo: 5.83%

De toda la capacidad aritmética de la GPU, solo se usa el 5.83%. Esto no es un error: con solo 3 bloques activos, el 90% de los SMs están apagados. Los que sí trabajan lo hacen al ~58% de su capacidad (porque el kernel alterna entre leer memoria y hacer cuentas).

#### Throughput de memoria: 5.56%

Solo el 5.56% del ancho de banda de DRAM está en uso (~18 GB/s de ~336 GB/s posibles). ¿Por qué tan poco? Porque los 192 KB que lee el kernel caben completamente en caché L1/L2. La DRAM casi no se toca.

#### Ocupación alcanzada: 16%

De los 48 slots de warps disponibles por SM, solo hay ~7.7 warps activos en promedio. Pero esto es engañoso: el promedio se calcula sobre los 30 SMs, y solo 3 tienen trabajo. En esos 3 SMs hay 1 bloque cada uno (8 warps), que es el 16% de 48.

El mensaje de `ncu` lo dice explícitamente:

```
WRN: This kernel grid is too small to fill the available resources,
     resulting in only 0.0 full waves across all SMs.
```

Una "wave" (ola) significa suficientes bloques para llenar todos los SMs. Con 30 SMs necesitas ≥30 bloques para 1 wave. Tienes 3. Cero waves completas.

#### Tasas de acierto de caché

| Caché | Tasa de acierto | Interpretación |
|-------|----------------|----------------|
| L1 | 57.8% | 6 de cada 10 accesos los sirve la L1 |
| L2 | 57.6% | De los que fallan en L1, 6 de cada 10 los sirve la L2 |

Estos números son **buenos para un dataset tan chico**. Los 192 KB caben en L2 (3 MB) e incluso parcialmente en L1 (128 KB por SM, pero cada bloque solo toca 64 KB).

#### Conclusión del perfil pequeño

La GPU es **excesiva** para este tamaño de problema. La versión secuencial en CPU tarda 0.4 ms — menos que el overhead de lanzar kernels y transferir datos. Esto no es una falla del código: es simplemente que no hay suficiente trabajo para mantener ocupada una GPU moderna.

---

## 5. El perfil del dataset grande (center_im, D=192)

### Configuración del kernel

```
Lanzamiento: <<<96 bloques, 256 hilos>>>
Matriz:      1,048,576 filas × 192 columnas
Datos leídos: 96 pares × 2 columnas × 1,048,576 filas × 8 bytes ≈ 1.5 GB
```

### Resultados y qué significan

#### Throughput de cómputo: 52%

La mitad de la capacidad aritmética está en uso. ¿Es bueno o malo? Depende. Si el cómputo estuviera al 100%, querría decir que la GPU no pierde tiempo esperando memoria. Si estuviera al 5%, querría decir que el código tiene poca aritmética. 52% es **normal para un kernel que alterna leer datos y hacer cuentas**.

#### Throughput de memoria: 89% (299 GB/s de 336 GB/s)

**Este es el número más importante del perfil.** Significa que la GPU está leyendo datos de VRAM al 89% de su velocidad máxima física. El bus de memoria de 192 bits a 7001 MHz tiene un pico teórico de ~336 GB/s. Alcanzar 299 GB/s sostenidos es **excelente** — indica que el patrón de acceso a memoria (*column-major*, lecturas coalescidas) es casi perfecto.

El mensaje de `ncu` lo confirma:

```
INF: The kernel is utilizing greater than 80.0% of the available
     compute or memory performance of the device.
```

"INF" no es error: es información. El kernel está usando >80% de la capacidad de memoria del dispositivo. Está completamente saturado.

#### Memory-bound: el veredicto

Cuando el throughput de memoria (89%) es mucho mayor que el de cómputo (52%), el kernel es **memory-bound**. Esto significa que **los accesos a DRAM son el cuello de botella**: los SMs pasan más tiempo esperando que lleguen datos de la VRAM que haciendo cuentas.

Para el Jacobi SVD esto es esperable. Cada barrido lee cada columna de la matriz D veces (una por cada par en que participa), y la mayor parte del "cómputo" son productos punto que son esencialmente lecturas acumuladas.

#### Ocupación alcanzada: 52.9%

Con 96 bloques, los 30 SMs están llenos. Cada SM tiene en promedio 25.4 warps activos de 48 posibles (53%). ¿Por qué no 100%?

La razón es que este kernel es *memory-bound*. Los warps pasan gran parte del tiempo **stalleados** (detenidos) esperando que la DRAM entregue los datos que pidieron. Mientras un warp espera, el SM puede ejecutar otro warp — pero solo si hay warps disponibles. Con 25.4 de 48, hay suficientes warps para mantener ocupados los núcleos de ejecución mientras otros esperan memoria. **Más warps no harían que el kernel fuera más rápido**, porque el límite no son los núcleos de cómputo sino la DRAM.

#### Tasas de acierto de caché

| Caché | Tasa de acierto | Interpretación |
|-------|----------------|----------------|
| L1 | 28.5% | Solo 3 de cada 10 accesos los sirve la L1 |
| L2 | 33.0% | De los que fallan en L1, 3 de cada 10 los sirve la L2 |

Estos números son **drásticamente menores** que en el dataset pequeño. La razón es simple: el conjunto de trabajo es enorme (~1.5 GB por lanzamiento del kernel). Las cachés suman apenas ~3 MB. Es imposible que quepa. La mayoría de los accesos van directo a DRAM, y por eso el ancho de banda de DRAM está al 89%.

---

## 6. Análisis de ocupación: ¿por qué teórica 100% pero alcanzada 53%?

`ncu` reporta dos números de ocupación:

### Ocupación teórica: 100%

Esto significa que el kernel **no tiene cuellos de botella de diseño** que limiten cuántos warps pueden coexistir en un SM. `ncu` lo calcula verificando tres límites:

| Recurso | Uso de nuestro kernel | Límite del hardware | ¿Es cuello de botella? |
|---------|----------------------|---------------------|------------------------|
| **Registros** | ~40 registros por hilo × 256 hilos = ~10K por bloque | 64K por SM | **Sí**: 6 bloques/SM máximo |
| **Shared memory** | 3 arrays × 256 doubles = 6 KB por bloque | 16-48 KB por SM | No: caben 14 bloques |
| **Warps** | 8 warps por bloque | 48 warps por SM | 48 ÷ 8 = 6 bloques |

Los tres límites convergen en 6 bloques por SM, que con 8 warps por bloque dan 48 warps = 100% ocupación teórica. Esto es un diseño balanceado: ningún recurso sobra ni falta.

### Ocupación alcanzada: 53%

La diferencia entre 100% (lo que *podrías* tener) y 53% (lo que *tienes*) se debe a que el kernel es *memory-bound*. Los warps se pasan la mayor parte del tiempo en estado *stalled* (detenidos) esperando datos de DRAM. `ncu` solo cuenta como "activos" los warps que están efectivamente ejecutando instrucciones.

En un kernel *memory-bound*, 53% es **saludable**. Si la ocupación fuera más alta (digamos 90%), significaría que los warps casi nunca esperan memoria — pero entonces el throughput de DRAM sería bajo, y el kernel sería *compute-bound*. Como nuestro límite real es la DRAM, tener ~53% de warps activos es suficiente para mantener el bus de memoria saturado.

---

## 7. ¿Por qué las cachés L1/L2 no ayudan más?

En el dataset chico, las cachés funcionan bien (~58% de aciertos). En el grande, caen a ~30%. Esto no es un problema del código: es **física**.

Cada lanzamiento del kernel para `center_im` lee ~1.5 GB de datos. La caché L2 completa de la RTX 3060 son 3 MB. Aunque la caché fuera perfecta y retuviera exactamente los datos que vas a necesitar, 3 MB es 0.2% de 1.5 GB. Es imposible que una caché de ese tamaño haga diferencia con un conjunto de trabajo tan grande.

La solución no es "mejorar las cachés" (no puedes), sino **diseñar el acceso a memoria para ser lo más eficiente posible cuando inevitablemente vas a DRAM**. Ahí es donde el layout *column-major* y los accesos coalescidos brillan: aunque cada acceso vaya a DRAM, se hacen en ráfagas de 32 direcciones consecutivas que el controlador de memoria puede servir eficientemente.

---

## 8. ¿Qué aprendimos del perfilado?

### Lo que funciona bien

1. **Acceso coalescido perfecto**: el layout *column-major* permite que los 32 hilos de un warp lean direcciones consecutivas. Esto es lo que permite alcanzar 299 GB/s (89% del pico).

2. **Uso balanceado de recursos**: los registros, la shared memory y los warps están dimensionados para que ningún recurso limite artificialmente la ocupación. La ocupación teórica del 100% es una señal de diseño limpio.

3. **Código correcto**: el error de reconstrucción (~$10^{-10}$ para datos reales) confirma que el kernel produce resultados numéricamente correctos.

### Lo que es inherente al algoritmo

4. **El kernel es memory-bound**: el Jacobi SVD lee cada columna $D$ veces por barrido. No hay forma de evitar esto sin cambiar de algoritmo. La bidiagonalización de LAPACK reutiliza mejor los datos en registros y caché, por eso es más rápida para matrices bien condicionadas.

5. **Datasets pequeños infrautilizan la GPU**: con D=6, solo 3 de 30 SMs trabajan. Esto no es un bug: simplemente no hay suficientes pares de columnas para mantener ocupada una GPU moderna. La solución práctica es no usar GPU para D < ~30.

### Implicancias para las otras implementaciones

6. **OpenMP sufre el mismo límite de memoria**: los hilos CPU compiten por el bus de memoria DDR4 (~50 GB/s), que es mucho más lento que la VRAM de la GPU (~336 GB/s). Por eso el speedup de OpenMP se estanca en ~1.78× con 4 hilos y degrada con más.

7. **LAPACK gana por mejor reuso de datos**: las rutinas BLAS nivel 3 de LAPACK operan sobre bloques de la matriz que caben en caché, reutilizando cada dato muchas veces antes de descartarlo. El Jacobi no puede hacer esto porque necesita acceder a columnas completas para cada par.

---

## 9. Cómo leer un perfil de ncu por tu cuenta

Si en el futuro quieres perfilar otro kernel, estos son los pasos:

### Paso 1: Revisa el mensaje de advertencia

```
WRN: kernel grid too small  →  lanzá más bloques o el problema es muy chico
INF: >80% utilization       →  la GPU está saturada, buen trabajo
```

### Paso 2: Compará memoria vs cómputo

```
Memory [%] > Compute [%]  →  memory-bound (optimizá accesos a memoria)
Compute [%] > Memory [%]  →  compute-bound (optimizá aritmética)
```

### Paso 3: Revisa la ocupación

```
Ocupación teórica < 100%  →  hay un cuello de botella de recursos (registros, shared memory)
Ocupación alcanzada ≪ teórica  →  memory-bound (warps esperando DRAM) o grid muy chico
```

### Paso 4: Interpretá las cachés

```
L1/L2 hit rate alta (>60%) + dataset chico    →  normal, datos caben en caché
L1/L2 hit rate baja (<35%) + memory-bound     →  esperable, conjunto de trabajo > caché
L1/L2 hit rate baja + compute-bound           →  algo raro, revisá el patrón de acceso
```

---

## 10. Glosario rápido

| Término | Significado |
|---------|-------------|
| **SM** | Streaming Multiprocessor. Unidad de ejecución en la GPU. La RTX 3060 tiene 30. |
| **Warp** | Grupo de 32 hilos que ejecutan la misma instrucción. Unidad real de ejecución. |
| **Ocupación** | Porcentaje de slots de warps que están ocupados. Teórica = lo que el código permite. Alcanzada = lo que realmente ocurre. |
| **Memory-bound** | El límite de velocidad es la memoria (DRAM). Los núcleos de cómputo esperan datos. |
| **Compute-bound** | El límite de velocidad es la aritmética. La memoria entrega datos más rápido de lo que se procesan. |
| **Coalescido** | Acceso a memoria donde hilos consecutivos leen direcciones consecutivas. Ideal para GPU. |
| **Shared memory** | Memoria rápida en el chip, compartida por todos los hilos de un bloque. Nosotros la usamos para la reducción en árbol. |
| **Registro** | El tipo de memoria más rápido. Cada hilo tiene sus propios registros (típicamente 32-64). |
| **Stalled** | Un warp detenido esperando que lleguen datos de memoria. |
| **ncu** | NVIDIA Nsight Compute. Perfilador de kernel individual. |
| **nsys** | NVIDIA Nsight Systems. Perfilador de línea de tiempo completa. |
