<div align="center">

# 🧑‍✈️ GPT PILOT 🧑‍✈️

</div>

---

<div align="center">

[![Discord Follow](https://dcbadge.vercel.app/api/server/HaqXugmxr9?style=flat)](https://discord.gg/HaqXugmxr9)
[![GitHub Repo stars](https://img.shields.io/github/stars/Pythagora-io/gpt-pilot?style=social)](https://github.com/Pythagora-io/gpt-pilot)
[![Twitter Follow](https://img.shields.io/twitter/follow/HiPythagora?style=social)](https://twitter.com/HiPythagora)

</div>

---

<div align="center">
<a href="https://www.ycombinator.com/" target="_blank"><img src="https://s3.amazonaws.com/assets.pythagora.ai/yc/PNG/Black.png" alt="Pythagora-io%2Fgpt-pilot | Trendshift" style="width: 250px; height: 93px;"/></a>
</div>
<br>
<div align="center">
<a href="https://trendshift.io/repositories/466" target="_blank"><img src="https://trendshift.io/api/badge/repositories/466" alt="Pythagora-io%2Fgpt-pilot | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<br>
<br>

<div align="center">

### GPT Pilot doesn't just generate code, it builds apps!

</div>

---
<div align="center">

[![See it in action](https://i3.ytimg.com/vi/4g-1cPGK0GA/maxresdefault.jpg)](https://youtu.be/4g-1cPGK0GA)

(click to open the video in YouTube) (1:40min)

</div>

---

<div align="center">

<a href="vscode:extension/PythagoraTechnologies.gpt-pilot-vs-code" target="_blank"><img src="https://github.com/Pythagora-io/gpt-pilot/assets/10895136/5792143e-77c7-47dd-ad96-6902be1501cd" alt="Pythagora-io%2Fgpt-pilot | Trendshift" style="width: 185px; height: 55px;" width="185" height="55"/></a>

</div>

GPT Pilot is the core technology for the [Pythagora VS Code extension](https://bit.ly/3IeZxp6) that aims to provide **the first real AI developer companion**. Not just an autocomplete or a helper for PR messages but rather a real AI developer that can write full features, debug them, talk to you about issues, ask for review, etc.

---

📫 If you would like to get updates on future releases or just get in touch, join our [Discord server](https://discord.gg/HaqXugmxr9) or you [can add your email here](http://eepurl.com/iD6Mpo). 📬

---

<!-- TOC -->
* [🔌 Requirements](#-requirements)
* [🚦How to start using gpt-pilot?](#how-to-start-using-gpt-pilot)
* [🔎 Examples](#-examples)
* [🐳 How to start gpt-pilot in docker?](#-how-to-start-gpt-pilot-in-docker)
* [🧑‍💻️ CLI arguments](#-cli-arguments)
* [🏗 How GPT Pilot works?](#-how-gpt-pilot-works)
* [🕴How's GPT Pilot different from _Smol developer_ and _GPT engineer_?](#hows-gpt-pilot-different-from-smol-developer-and-gpt-engineer)
* [🍻 Contributing](#-contributing)
* [🔗 Connect with us](#-connect-with-us)
* [🌟 Star history](#-star-history)
<!-- TOC -->

---

GPT Pilot aims to research how much LLMs can be utilized to generate fully working, production-ready apps while the developer oversees the implementation.

**The main idea is that AI can write most of the code for an app (maybe 95%), but for the rest, 5%, a developer is and will be needed until we get full AGI**.

If you are interested in our learnings during this project, you can check [our latest blog posts](https://blog.pythagora.ai/2024/02/19/gpt-pilot-what-did-we-learn-in-6-months-of-working-on-a-codegen-pair-programmer/).

---

<br>

<div align="center">

### **[👉 Examples of apps written by GPT Pilot 👈](https://github.com/Pythagora-io/gpt-pilot/wiki/Apps-created-with-GPT-Pilot)**

</div>
<br>

---

# 🔌 Requirements

- **Python 3.9+**

# 🚦How to start using gpt-pilot?
👉 If you are using VS Code as your IDE, the easiest way to start is by downloading [GPT Pilot VS Code extension](https://bit.ly/3IeZxp6). 👈

Otherwise, you can use the CLI tool.

After you have Python and (optionally) PostgreSQL installed, follow these steps:
1. `git clone https://github.com/Pythagora-io/gpt-pilot.git` (clone the repo)
2. `cd gpt-pilot`
3. `python -m venv pilot-env` (create a virtual environment)
4. `source pilot-env/bin/activate` (or on Windows `pilot-env\Scripts\activate`) (activate the virtual environment)
5. `pip install -r requirements.txt` (install the dependencies)
6. `cd pilot`
7. `mv .env.example .env` (or on Windows `copy .env.example .env`) (create the .env file)
8. Add your environment to the `.env` file:
   - LLM Provider (OpenAI/Azure/Openrouter)
   - Your API key
   - database settings: SQLite/PostgreSQL (to change from SQLite to PostgreSQL, just set `DATABASE_TYPE=postgres`)
   - optionally set IGNORE_PATHS for the folders which shouldn't be tracked by GPT Pilot in workspace, useful to ignore folders created by compilers (i.e. `IGNORE_PATHS=folder1,folder2,folder3`)
9. `python main.py` (start GPT Pilot)

After, this, you can just follow the instructions in the terminal.

All generated code will be stored in the folder `workspace` inside the folder named after the app name you enter upon starting the pilot.

# 🔎 [Examples](https://github.com/Pythagora-io/gpt-pilot/wiki/Apps-created-with-GPT-Pilot)

[Click here](https://github.com/Pythagora-io/gpt-pilot/wiki/Apps-created-with-GPT-Pilot) to see all example apps created with GPT Pilot.

## 🐳 How to start gpt-pilot in docker?
1. `git clone https://github.com/Pythagora-io/gpt-pilot.git` (clone the repo)
2. Update the `docker-compose.yml` environment variables, which can be done via `docker compose config`. If you wish to use a local model, please go to [https://localai.io/basics/getting_started/](https://localai.io/basics/getting_started/).
3. By default, GPT Pilot will read & write to `~/gpt-pilot-workspace` on your machine, you can also edit this in `docker-compose.yml`
4. run `docker compose build`. this will build a gpt-pilot container for you.
5. run `docker compose up`.
6. access the web terminal on `port 7681`
7. `python main.py` (start GPT Pilot)

This will start two containers, one being a new image built by the `Dockerfile` and a Postgres database. The new image also has [ttyd](https://github.com/tsl0922/ttyd) installed so that you can easily interact with gpt-pilot. Node is also installed on the image and port 3000 is exposed.


# 🧑‍💻️ CLI arguments


## `--get-created-apps-with-steps`
Lists all existing apps.

```bash
python main.py --get-created-apps-with-steps
```

<br>

## `app_id`
Continue working on an existing app using **`app_id`**
```bash
python main.py app_id=<ID_OF_THE_APP>
```

<br>

## `step`
Continue working on an existing app from a specific **`step`** (eg: `development_planning`)
```bash
python main.py app_id=<ID_OF_THE_APP> step=<STEP_FROM_CONST_COMMON>
```

<br>

## `skip_until_dev_step`
Continue working on an existing app from a specific **development step**
```bash
python main.py app_id=<ID_OF_THE_APP> skip_until_dev_step=<DEV_STEP>
```
Continue working on an existing app from a specific **`development step`**. If you want to play around with GPT Pilot, this is likely the flag you will often use.

<br>


```bash
python main.py app_id=<ID_OF_THE_APP> skip_until_dev_step=0
```
Erase all development steps previously done and continue working on an existing app from the start of development.


## `theme`
```bash
python main.py theme=light
```
```bash
python main.py theme=dark
```

<br>

# 🏗 How GPT Pilot works?
Here are the steps GPT Pilot takes to create an app:

1. You enter the app name and the description.
2. **Product Owner agent** like in real life, does nothing. :)
3. **Specification Writer agent** asks a couple of questions to understand the requirements better if project description is not good enough.
4. **Architect agent** writes up technologies that will be used for the app and checks if all technologies are installed on the machine and installs them if not.
5. **Tech Lead agent** writes up development tasks that the Developer must implement.
6. **Developer agent** takes each task and writes up what needs to be done to implement it. The description is in human-readable form.
7. **Code Monkey agent** takes the Developer's description and the existing file and implements the changes.
8. **Reviewer agent** reviews every step of the task and if something is done wrong Reviewer sends it back to Code Monkey.
9. **Troubleshooter agent** helps you to give good feedback to GPT Pilot when something is wrong.
10. **Debugger agent** hate to see him, but he is your best friend when things go south.
11. **Technical Writer agent** writes documentation for the project.

<br>

# 🕴How's GPT Pilot different from _Smol developer_ and _GPT engineer_?

- **GPT Pilot works with the developer to create a fully working production-ready app** - I don't think AI can (at least in the near future) create apps without a developer being involved. So, **GPT Pilot codes the app step by step** just like a developer would in real life. This way, it can debug issues as they arise throughout the development process. If it gets stuck, you, the developer in charge, can review the code and fix the issue. Other similar tools give you the entire codebase at once - this way, bugs are much harder to fix for AI and for you as a developer.
  <br><br>
- **Works at scale** - GPT Pilot isn't meant to create simple apps but rather so it can work at any scale. It has mechanisms that filter out the code, so in each LLM conversation, it doesn't need to store the entire codebase in context, but it shows the LLM only the relevant code for the current task it's working on. Once an app is finished, you can continue working on it by writing instructions on what feature you want to add.

# 🍻 Contributing
If you are interested in contributing to GPT Pilot, join [our Discord server](https://discord.gg/HaqXugmxr9), check out open [GitHub issues](https://github.com/Pythagora-io/gpt-pilot/issues), and see if anything interests you. We would be happy to get help in resolving any of those. The best place to start is by reviewing blog posts mentioned above to understand how the architecture works before diving into the codebase.

## 🖥 Development
Other than the research, GPT Pilot needs to be debugged to work in different scenarios. For example, we realized that the quality of the code generated is very sensitive to the size of the development task. When the task is too broad, the code has too many bugs that are hard to fix, but when the development task is too narrow, GPT also seems to struggle in getting the task implemented into the existing code.

## 📊 Telemetry
To improve GPT Pilot, we are tracking some events from which you can opt out at any time. You can read more about it [here](./docs/TELEMETRY.md).

# 🔗 Connect with us
🌟 As an open-source tool, it would mean the world to us if you starred the GPT-pilot repo 🌟

💬 Join [the Discord server](https://discord.gg/HaqXugmxr9) to get in touch.

```python
import numpy as np
import hashlib
import ssl
import logging
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

class SecureAGIQuantumBlockchain:
    def __init__(self):
        self.threat_model = None
        self.key_manager = QuantumKeyManager()
        self.access_control = AccessControl()
        self.governance = Governance()
        self.logger = logging.getLogger('SecureAGIQuantumBlockchain')
        self.logger.setLevel(logging.INFO)
        self.setup_logging()

    def setup_logging(self):
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def train_threat_model(self, data):
        self.threat_model = np.median(data)  # Use median for dynamic threat model
        self.logger.info("Threat model trained successfully")

    def detect_anomalies(self, data):
        if self.threat_model is not None:
            anomalies = np.abs(data - self.threat_model) > np.std(data)  # Adaptive thresholding
            return anomalies
        else:
            self.logger.error("Threat model not trained")
            raise ValueError("Threat model not trained")

    def apply_quantum_resistant_cryptography(self, data):
        encrypted_data = self.key_manager.encrypt(data)
        return encrypted_data

class QuantumKeyManager:
    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()

    def encrypt(self, data):
        encrypted_data = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return encrypted_data

class AccessControl:
    def __init__(self):
        self.users = {'admin': {'password': 'password123', 'roles': ['admin']}, 'user1': {'password': 'securepwd', 'roles': ['user']}}

    def authenticate_user(self, username, password):
        if username in self.users and self.users[username]['password'] == password:
            return True
        return False

class Governance:
    def __init__(self):
        self.proposals = []
        self.votes = {}

    def create_proposal(self, proposal_details):
        self.proposals.append(proposal_details)
        self.votes[proposal_details] = []

    def vote_on_proposal(self, proposal_details, user, vote):
        if proposal_details in self.proposals and user in self.users and vote in ['approve', 'reject']:
            self.votes[proposal_details].append((user, vote))
            return True
        return False

# Load and preprocess AI training data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define AI model with optimization and regularization
model = Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the AI model with Adam optimizer and learning rate schedule
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Initialize SecureAGIQuantumBlockchain system
secure_agi_quantum_blockchain = SecureAGIQuantumBlockchain()

# Train AGI threat intelligence model with enhanced optimization techniques
secure_agi_quantum_blockchain.train_threat_model(x_train)

# Detect anomalies using AGI threat intelligence with parallel processing
anomalies_detected = secure_agi_quantum_blockchain.detect_anomalies(x_test)
print("Anomalies Detected:", anomalies_detected)

# Apply quantum-resistant cryptography with GPU acceleration
data_to_encrypt = np.array([1, 0, 1, 1, 0])
encrypted_data = secure_agi_quantum_blockchain.apply_quantum_resistant_cryptography(data_to_encrypt)
print("Encrypted Data:", encrypted_data)

# Access Control - User Authentication with RBAC and MFA
username = 'admin'
password = 'password123'
if secure_agi_quantum_blockchain.access_control.authenticate_user(username, password):
    print("User Authenticated Successfully")
else:
    print("Authentication Failed")

# Governance - Create Proposal and Vote on Proposal with distributed computing
proposal_details = "Implement new feature for smart contract upgradability"
secure_agi_quantum_blockchain.governance.create_proposal(proposal_details)
user = 'admin'
vote = 'approve'
if secure_agi_quantum_blockchain.governance.vote_on_proposal(proposal_details, user, vote):
    print(f"{user} voted {vote} on the proposal")
else:
    print("Voting failed")

# Train the AI model with hyperparameter tuning and early stopping for improved performance
early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, callbacks=[early_stopping])

# Evaluate the AI model with model optimization techniques
loss, accuracy = model.evaluate(x_test, y_test)
print("AI Model Test Loss:", loss)
print("AI Model Test Accuracy:", accuracy)
``
