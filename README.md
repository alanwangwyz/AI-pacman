# Pacman Project 
![](https://img.shields.io/badge/Game-Pacman-green.svg)
![](https://img.shields.io/badge/Strategy-ApproximateQ&A*-blue.svg)
![](https://img.shields.io/badge/Language-python2-orange.svg)

![image](https://github.com/alanwangwyz/AI-pacman/blob/master/image/facebook-messenger-pac-man.png)

## Acknowledgement ##
🔐`Berkeley Uni CS188`

## Strategy ##
👉`Approximate Q learning`algorithm to adjust pacman's movement as keep following the maxmimum value updated by the formula
_weights_ and _features_ to do the update

__`chaseEnemyValue`, `capsuleValue`, `successorScore`__

👉`A\*` to escape when enemy is approaching e.g. within 6 blocks



## Experiment Results ##
| Score | Win | Lost | Use Heuristic | Basic | Medium | Top |
| :---: | :--:| :--: | :-----------: |   :-: |  :-:   | :-: |
|  160  |  51 |  63  |      No       |   Yes |  No    |  No |
|  158  |  46 |  75  |      No       |   Yes |  No    |  No |
|  254  |  70 |  49  |      Yes      |   Yes |  Yes   |  Yes |
