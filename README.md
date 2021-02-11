The code can be run only on python 2.7. Else, creating a virtual environment can work.
The code for the MCTSagent can be found in multiagents.py

Instructions to run the code:
For MCTS:
on smallClassic:
	python pacman.py -p MCTSAgent -l smallClass
on testClassic:
	python pacman.py -p MCTSAgent -l testClass
on mediumCLassic:
	python pacman.py -p MCTSAgent -l meduimClassic 

For MinimaxAgent:
on smallClassic:
	python pacman.py -p MinimaxAgent -l smallClass
on testClassic:
	python pacman.py -p MinimaxAgent -l testClass
on mediumCLassic:
	python pacman.py -p MinimaxAgent -l meduimClassic 

For AlphaBetaAgent:
on smallClassic:
	python pacman.py -p AlphaBetaAgent -l smallClass
on testClassic:
	python pacman.py -p AlphaBetaAgent -l testClass
on mediumCLassic:
	python pacman.py -p AlphaBetaAgent -l meduimClassic 
