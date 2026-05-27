# VLM Server Requirements - LeRobot BT Stack

Questo documento elenca TUTTI i requisiti che un server VLM esterno deve
rispettare per integrarsi correttamente con lo stack Behavior Tree di LeRobot.
E pensato per essere dato a un altro agente AI che deve implementare il lato
VLM senza dover leggere l'intero codice sorgente.

---

## 1. TOPIC ROS2 DA USARE

Il server VLM deve interagire con due topic ROS2 (entrambi di tipo
`std_msgs/String` contenenti JSON):

| Direzione | Topic (default) | Scopo |
|-----------|----------------|-------|
| IN (ascolto) | `/lerobot_bt/vlm_request` | Il BT stack pubblica qui quando serve un verdetto VLM |
| OUT (pubblicazione) | `/lerobot_bt/vlm_result` | Il server VLM pubblica qui il risultato della verifica |

NOTE: I nomi dei topic sono configurabili nell'executor YAML (campi
`vlm_request_topic` e `vlm_result_topic`). Il server VLM dovrebbe
permettere di cambiarli via parametri ROS2.

---

## 2. FORMATO RICHIESTA VLM (topic `/lerobot_bt/vlm_request`)

Quando il BT ha bisogno di un verdetto VLM, pubblica un JSON con questa
struttura esatta:

```json
{
  "event": "vlm_check_requested",
  "skill_name": "place_first_toast",
  "attempt_id": 3,
  "status": "PENDING",
  "message": "Awaiting VLM result for skill 'place_first_toast'.",
  "allowed_statuses": [
    "PENDING",
    "RUNNING",
    "WAIT_HUMAN",
    "MANUAL_INTERVENTION_REQUIRED",
    "SUCCESS",
    "FAILURE"
  ],
  "allowed_next_actions": [
    "CONTINUE",
    "RETRY_SKILL",
    "WAIT_HUMAN",
    "REQUEST_MANUAL_INTERVENTION"
  ]
}
```

### Campi obbligatori nella richiesta:

| Campo | Tipo | Significato |
|-------|------|-------------|
| `skill_name` | `string` | Nome del checkpoint BT corrente (es. `place_first_toast`, `initial_scene_ready`, `make_sandwich.task_complete`) |
| `attempt_id` | `int` | Contatore monotonicamente crescente per ogni tentativo di quella skill. Deve essere rimandato indietro nella risposta |
| `status` | `string` | Sempre `PENDING` nelle richieste |
| `message` | `string` | Messaggio human-readable di contesto |
| `allowed_statuses` | `list[string]` | I valori accettati in risposta - usa solo questi |
| `allowed_next_actions` | `list[string]` | I valori accettati per il campo `next_action` nella risposta |

---

## 3. FORMATO RISPOSTA VLM (topic `/lerobot_bt/vlm_result`)

Il server VLM deve pubblicare un JSON su `/lerobot_bt/vlm_result` con almeno
questi campi. Ci sono DUE modalita di risposta:

### Modalita A: Risposta con `status` (semplice, consigliata)

```json
{
  "skill_name": "place_first_toast",
  "attempt_id": 3,
  "status": "SUCCESS",
  "message": "The toast is correctly placed on the sandwich."
}
```

### Modalita B: Risposta con `next_action` (per flussi piu complessi)

```json
{
  "skill_name": "place_first_toast",
  "attempt_id": 3,
  "next_action": "RETRY_SKILL",
  "message": "Toast fell off the table. Robot should retry."
}
```

### Campi della risposta:

| Campo | Tipo | Obbligatorio | Significato |
|-------|------|-------------|-------------|
| `skill_name` | `string` | SI | Deve coincidere esattamente con quello ricevuto nella richiesta |
| `attempt_id` | `int` | SI | Deve coincidere con l`attempt_id` della richiesta attiva. Usare `0` per applicare al piu recente |
| `status` | `string` | SI* | Uno dei valori in `allowed_statuses` (vedi sezione 4) |
| `next_action` | `string` | SI* | Alternativa a `status`, uno dei valori in `allowed_next_actions` (vedi sezione 4) |
| `message` | `string` | No | Spiegazione human-readable del verdetto |
| `scene_ready` | `bool` | No | Scorciatoia: `true` -> SUCCESS |
| `human_help_required` | `bool` | No | Scorciatoia: `true` -> MANUAL_INTERVENTION_REQUIRED |
| `anomaly_detected` | `bool` | No | Scorciatoia: `true` -> FAILURE |
| `scene_id` | `string` | No | Se inizia con `SCENE_` e finisce con `_READY` -> SUCCESS; se `TASK_COMPLETE` -> SUCCESS; se `ANOMALY_DETECTED` -> FAILURE; se `HUMAN_HELP_REQUIRED` -> MANUAL_INTERVENTION_REQUIRED |

* Almeno uno tra `status` e `next_action` deve essere presente.

---

## 4. VOCABOLARIO ESATTO DEGLI STATI

### 4.1 Valori di `status` (usa ESATTAMENTE queste stringhe, case-sensitive)

| Status | Effetto sul BT | Quando usarlo |
|--------|---------------|---------------|
| `PENDING` | Il BT aspetta (RUNNING). Il check non e ancora partito. | Subito dopo aver ricevuto la richiesta, mentre il VLM sta caricando il modello |
| `RUNNING` | Il BT aspetta (RUNNING). Il VLM sta processando le immagini. | Durante l'inferenza VLM |
| `WAIT_HUMAN` | Il BT aspetta (RUNNING). Serve un intervento umano. | Quando il VLM non puo decidere e serve un operatore |
| `MANUAL_INTERVENTION_REQUIRED` | Il BT aspetta (RUNNING). L'operatore deve intervenire fisicamente. | Scena in stato non recuperabile automaticamente |
| `SUCCESS` | Il BT avanza al nodo successivo. | La scena e corretta, il task e completato |
| `FAILURE` | Il BT ritenta la skill (via `RetryUntilSuccessful`). | La scena non e corretta, il robot deve riprovare |

### 4.2 Valori di `next_action` (alternativa a `status`)

| next_action | Equivalente a `status` | Significato |
|-------------|------------------------|-------------|
| `CONTINUE` | `SUCCESS` | Avanza al prossimo passo |
| `PROCEED` | `SUCCESS` | Alias di CONTINUE |
| `RETRY` | `FAILURE` | Ritenta la skill corrente |
| `RETRY_SKILL` | `FAILURE` | Ritenta la skill corrente |
| `WAIT` | `RUNNING` | Il VLM sta ancora processando |
| `WAIT_HUMAN` | `WAIT_HUMAN` | Serve intervento umano |
| `REQUEST_MANUAL_INTERVENTION` | `MANUAL_INTERVENTION_REQUIRED` | Serve intervento manuale |
| `MANUAL_INTERVENTION` | `MANUAL_INTERVENTION_REQUIRED` | Alias |

### 4.3 Valori speciali per `scene_id`

Il campo `scene_id` viene normalizzato automaticamente:
- Pattern `SCENE_*_READY` -> `SUCCESS` (es. `SCENE_0_READY`, `SCENE_INITIAL_READY`)
- `TASK_COMPLETE` -> `SUCCESS`
- `ANOMALY_DETECTED` -> `FAILURE`
- `HUMAN_HELP_REQUIRED` -> `MANUAL_INTERVENTION_REQUIRED`

---

## 5. MACCHINA A STATI DEI TENTATIVI

Il BT stack tiene traccia di ogni tentativo con un `attempt_id` monotonicamente
crescente per skill. Il server VLM NON deve tenere uno stato interno
complesso, ma deve rispettare queste regole:

### Regola 1: Accoppiamento `skill_name` + `attempt_id`
- Ogni risposta VLM DEVE contenere lo stesso `skill_name` e `attempt_id`
  ricevuti nella richiesta.
- Se l`attempt_id` nella risposta non corrisponde piu all'ultimo tentativo
  attivo, la risposta viene scartata silenziosamente (il robot ha gia
  ricominciato).

### Regola 2: Stati terminali bloccano ulteriori update
- Una volta che una skill ha ricevuto `SUCCESS` o `FAILURE`, eventuali
  ulteriori risposte VLM per lo stesso tentativo vengono ignorate.

### Regola 3: Timeout VLM
- Il campo `vlm_timeout_s` nell'executor YAML (default: `0.0` che significa
  nessun timeout) definisce dopo quanti secondi un tentativo in stato
  `PENDING`/`RUNNING`/`WAIT_HUMAN`/`MANUAL_INTERVENTION_REQUIRED` viene
  automaticamente marcato come `FAILURE`.
- Il server VLM non deve implementare il timeout - lo gestisce il BT stack.
- Nota: con `vlm_timeout_s: 0.0` il BT aspetta indefinitamente.

### Regola 4: Live stop di una skill in esecuzione
- Se arriva un verdetto VLM (`SUCCESS`, `FAILURE`, `WAIT_HUMAN`, o
  `MANUAL_INTERVENTION_REQUIRED`) mentre una policy e ancora in esecuzione
  sul robot, il BT stack ferma immediatamente la policy e applica il verdetto.
- Questo permette a un operatore umano o a un VLM di interrompere una skill
  che sta chiaramente fallendo.

---

## 6. FLUSSO TEMPORALE COMPLETO (ESEMPIO: `make_sandwich`)

```
1. BT C++ invia comando "place_first_toast" via servizio ROS2
2. Python server esegue la policy sul robot
3. Policy finisce -> Python server apre un VLM check attempt (attempt_id: 1)
4. Python server pubblica su /lerobot_bt/vlm_request:
   {"skill_name": "place_first_toast", "attempt_id": 1, "status": "PENDING", ...}
5. QUI INTERVIENE IL SERVER VLM
   a. Riceve la richiesta
   b. Opzionale: pubblica status RUNNING (se ci mette tempo)
   c. Analizza le immagini dalle camere
   d. Decide SUCCESS o FAILURE
   e. Pubblica su /lerobot_bt/vlm_result:
      {"skill_name": "place_first_toast", "attempt_id": 1, "status": "SUCCESS"}
6. BT C++ riceve SUCCESS -> avanza al prossimo nodo
7. Se invece il VLM risponde FAILURE -> BT C++ ritenta place_first_toast
   (nuovo attempt_id: 2)
```

### Pubblicazioni intermedie consentite

Il server VLM puo pubblicare aggiornamenti intermedi per indicare che sta
lavorando:

```json
{"skill_name": "place_first_toast", "attempt_id": 1, "status": "RUNNING"}
```

```json
{"skill_name": "place_first_toast", "attempt_id": 1, "status": "WAIT_HUMAN"}
```

Questi update tengono il BT in stato RUNNING (non bloccano, ma non
avanzano). Solo `SUCCESS` o `FAILURE` cambiano lo stato del BT.

---

## 7. GESTIONE DELLE IMMAGINI

Il server VLM deve accedere alle immagini dalle camere del robot. Ci sono
due modi:

### Opzione A: Sottoscrizione ai topic ROS2 delle camere
- I topic camera sono pubblicati dal nodo `lerobot_bt_skill_server`
- I nomi dipendono dalla configurazione robot (es. topic per camera `wrist`,
  camera `left`)
- Il server VLM deve conoscere i nomi dei topic o riceverli come parametri

### Opzione B: Chiamata diretta al robot (solo se roadable)
- Usare l'API LeRobot per catturare un'osservazione corrente
- Questo richiede che il server VLM sia nello stesso processo o abbia
  accesso all'oggetto robot

---

## 8. INTERFACCIAMENTO CON PANDA LIVE CAMERA (OPZIONALE)

Se si usa il bridge `lerobot-bt-vlm-bridge` (nodo `bt_vlm_bridge.py`), la
comunicazione e semplificata a stringhe semplici:

| Direzione | Topic | Formato |
|-----------|-------|---------|
| BT -> Bridge -> VLM | `/panda/vlm/request` | `std_msgs/String` con prompt testuale |
| VLM -> Bridge -> BT | `/panda/vlm/status` | `std_msgs/String` con uno di: `SUCCESS`, `FAILED`, `FAILURE`, `STILL_RUNNING`, `RUNNING`, `PENDING` |

Il bridge traduce automaticamente tra il protocollo JSON del BT e il
protocollo a stringhe semplici del Panda VLM. Il server VLM in questo
caso deve solo:
1. Ascoltare `/panda/vlm/request`
2. Pubblicare su `/panda/vlm/status`

Il formato della richiesta e un template configurabile (parametro
`command_template`), default:

```
Verify whether the current robot task/check is complete.
BT check name: {skill_name}
Attempt id: {attempt_id}
BT message: {message}
Return exactly one token: SUCCESS, FAILED, or STILL_RUNNING.
```

---

## 9. REQUISITI TECNICI PER IL SERVER VLM

### 9.1 ROS2
- Deve essere un nodo ROS2 (Python o C++)
- Deve usare `std_msgs/String` per entrambi i topic
- Deve inizializzare `rclpy` (Python) o `rclcpp` (C++)
- Consigliato: usare un `MultiThreadedExecutor` se il VLM e bloccante

### 9.2 Robustezza
- Validazione JSON: scartare messaggi malformati senza crashare
- skill_name vuoto: non pubblicare risposte con `skill_name` vuoto
- attempt_id mancante: assumere `0` (si applica all'ultimo tentativo)
- Connessione persa: riconnettersi automaticamente ai topic
- Idempotenza: pubblicare lo stesso risultato piu volte non deve
  causare problemi (il BT stack ignora duplicati)

### 9.3 Performance
- Il VLM dovrebbe rispondere entro un tempo ragionevole (idealmente < 5s)
- Per VLM lenti, pubblicare `RUNNING` come acknowledgment immediato
- Usare un meccanismo di cancellazione: se arriva una nuova richiesta per
  la stessa skill con `attempt_id` piu alto, abortire l'inferenza precedente

---

## 10. CHECKLIST RIASSUNTIVA PER L'IMPLEMENTAZIONE

- [ ] Il nodo ROS2 si chiama `lerobot_bt_vlm_server` (o nome configurabile)
- [ ] Si sottoscrive a `/lerobot_bt/vlm_request` (`std_msgs/String`)
- [ ] Pubblica su `/lerobot_bt/vlm_result` (`std_msgs/String`)
- [ ] I nomi dei topic sono parametri ROS2 (`vlm_request_topic`, `vlm_result_topic`)
- [ ] Legge correttamente il JSON in arrivo (campi: `skill_name`, `attempt_id`, `status`, `message`)
- [ ] Pubblica risposte JSON con almeno `skill_name`, `attempt_id`, e uno tra `status`/`next_action`
- [ ] `skill_name` e `attempt_id` nella risposta corrispondono ESATTAMENTE alla richiesta
- [ ] Usa solo i valori di status autorizzati: `PENDING`, `RUNNING`, `WAIT_HUMAN`, `MANUAL_INTERVENTION_REQUIRED`, `SUCCESS`, `FAILURE`
- [ ] Pubblica `RUNNING` (o `PENDING`) subito dopo aver ricevuto la richiesta, se l'inferenza e lenta
- [ ] Quando arriva una nuova richiesta con `attempt_id` piu alto, abortisce l'inferenza in corso
- [ ] Non crasha se riceve JSON malformato
- [ ] Non crasha se `skill_name` e vuoto
- [ ] Ha accesso alle immagini dalle camere (parametro per i topic camera o API robot)
- [ ] (Opzionale) Supporta il flag `human_help_required: true` per richiedere intervento umano
- [ ] (Opzionale) Supporta il campo `next_action` come alternativa a `status`
- [ ] (Opzionale) Compatibile con il bridge `lerobot-bt-vlm-bridge` per il protocollo Panda semplificato
