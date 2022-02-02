# RL-Poker

This project uses multiple reinforcement learning algorithms to train agents for the holdem texa version of the poker game.

<br />
<div id="top"></div>
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#agents">Agents</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

![image](https://user-images.githubusercontent.com/62221313/152180551-06d84750-01b7-4a6b-8cb0-0d7f16199e4a.png)

In this project we explored the basic reinformcent learning arlogirtms (SARSA, Expected SARSA, Q Learning) and some based on neural networks (DQN and its variants) to see how each algorithm would perform in a game of poker. Our objective was to find the best model with whom to play the game.

The models were trained agains the random model which comes with the enviroment, to avoid the possibility of them to learn the patter of their adversary rather than developing a true strategy for the game. Each model has its own logic and learning mechanism on which it bases its behavior.

<p align="right">(<a href="#top">back to top</a>)</p>



### Agents

* SARSA
* Expected SARSA
* Q Learning
* DQN Base
* DQN Target Network 
* DQN Target Network and Experience Replay
* DQN All

For more details regarding each agent please refer to <a href="https://github.com/claudia-maria-dudau/RL-Poker"><strong>the docs »</strong></a>

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Usage

You need to have python 3 and pip 3 already installed on your machine.


### Installation

1. Clone the repo  
   `git clone https://github.com/claudia-maria-dudau/RL-Poker.git`

2. cd into this project  

3. Install the environment from github  
   `pip install -e git+https://github.com/dickreuter/neuron_poker#egg=neuron_poker`
   
4. Install necessary libraries  
   `pip install -r requirements.txt`
   
5. Run project  
   `pythom main.py [option]`

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Project Link: [https://github.com/claudia-maria-dudau/RL-Poker](https://github.com/claudia-maria-dudau/RL-Poker)

This project was made by:
 - Agha Mara
 - Buduroes Bianca
 - Dudau Claudia
 - Poinarita Diana

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Poker Environment](https://github.com/dickreuter/neuron_poker)

<p align="right">(<a href="#top">back to top</a>)</p>
