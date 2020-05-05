# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # Make sure to use the Stack , Queue and PriorityQueue data
    # structures provided to you in util.py
    from util import Stack

    # visited = [False for _ in range(10000000) # test range
    visited = []  # puts visited nodes inside visited

    # if initial goal state and initial start state is the same, return empty list
    if problem.isGoalState(problem.getStartState()):
        return []

    s = Stack()  # create a queue
    s.push((problem.getStartState(), []))  # push the start position and the empty path to the queue

    while s:
        new_pos, path = s.pop()  # get the values inside the queue by popping list
        # print(path)
        visited.append(new_pos)  # just append new_pos instead of finding whether visited in new_pos is T/F

        # if it is visited, continue
        # if visited[new_pos]:   # can't do this because visited[A] is a str
        #     continue

        # if it is not visited, make it true
        # else:
        #     visited[new_pos] = True    # can't do this either b/c it's str

        # if the position is the same as the goal state, return the path of that pos
        # if new_pos == problem.isGoalState():   # can't do this cuz goalstate is bool
        #     return path

        # print(path, 'path')
        # print(visited, 'visited')
        if problem.isGoalState(new_pos):  # if this prints out true, new_pos is at the goal state
            return path

        # need to get successor values to go to the next pos
        successor = problem.getSuccessors(new_pos)
        # In the case that there are no successors, just pop

        # print(successor, 'successor of ', new_pos)

        if successor:
            for i in successor:
                # print(i) # ('B', '0:A->B', 1.0), ('C', '1:A->C', 2.0), ('D', '2:A->D', 4.0)
                # print(i[0]) # B, C, D. This will be the new_pos values
                # print(i[1]) # 0:A->B, 1:A->C, 2:A->D. This will become the path of A
                if i[0] not in visited:  # check if successors are already visited. If it is, no need to repeat
                    # push i[0] and i[1] to queue
                    s.push((i[0], path + [i[1]]))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    # from collections import deque

    # very similar to DFS
    visited = [] # check if the node is visited

    # if initial goal state and initial start state is the same, return empty list
    if problem.isGoalState(problem.getStartState()):
        return []

    q = Queue() # create a queue
    q.push((problem.getStartState(), [])) # push the start position and the empty path to the queue
    # print(q)

    while q:
        new_pos, path = q.pop() # get the values inside the queue by popping list
        # print(path)

        # in the output below, A -> B1, C, B2. But B1 -> C. But C is already visited so we skip that one and go next
        # this should fix my error
        if new_pos not in visited:
            visited.append(new_pos) # just append new_pos instead of finding whether visited in new_pos is T/F
        else:
            continue

        # print(path, 'path')
        # print(visited, 'visited')
        if problem.isGoalState(new_pos): # if this prints out true, new_pos is at the goal state
            return path

        # need to get successor values to go to the next pos
        successor = problem.getSuccessors(new_pos)
        # print(successor, 'successor')
        if successor:
            for i in successor:
                if i[0] not in visited: # check if successors are already visited. If it is, no need to repeat
                    # push i[0] and i[1] to queue
                    q.push((i[0], path + [i[1]])) # appends left of the list. self.list.insert(0,item)

        # getting an error
        # ** *student solution: ['1:A->C', '0:C->D', '1:D->F', '0:F->G']
        # ** *student expanded_states: ['A', 'B1', 'C', 'B2', 'C', 'D', 'D', 'E1', 'F', 'E2', 'E1', 'F', 'E2', 'F']

        # ** *correct solution: ['1:A->C', '0:C->D', '1:D->F', '0:F->G']
        # ** *correct expanded_states: ['A', 'B1', 'C', 'B2', 'D', 'E1', 'F', 'E2']
        # ** *correct rev_solution: ['1:A->C', '0:C->D', '1:D->F', '0:F->G']
        # ** *correct rev_expanded_states: ['A', 'B2', 'C', 'B1', 'D', 'E2', 'F', 'E1']

        # in the expanded_state, it is repeatedly going to already visited states for some reason

        # output
        # [] path
        # ['A'] visited
        # [('B1', '0:A->B1', 1.0), ('C', '1:A->C', 2.0), ('B2', '2:A->B2', 4.0)] successor
        # ['0:A->B1'] path
        # ['A', 'B1'] visited
        # [('C', '0:B1->C', 8.0)] successor
        # ['1:A->C'] path
        # ['A', 'B1', 'C'] visited
        # [('D', '0:C->D', 32.0)] successor
        # ['2:A->B2'] path
        # ['A', 'B1', 'C', 'B2'] visited
        # [('C', '0:B2->C', 16.0)] successor
        # ['0:A->B1', '0:B1->C'] path
        # ['A', 'B1', 'C', 'B2', 'C'] visited      WHY DOES IT APPEND C WHEN C IS ALREADY IN LIST???
        # [('D', '0:C->D', 32.0)] successor
        # ['1:A->C', '0:C->D'] path
        # ['A', 'B1', 'C', 'B2', 'C', 'D'] visited
        # [('E1', '0:D->E1', 64.0), ('F', '1:D->F', 128.0), ('E2', '2:D->E2', 256.0)] successor
        # ['0:A->B1', '0:B1->C', '0:C->D'] path
        # ['A', 'B1', 'C', 'B2', 'C', 'D', 'D'] visited
        # [('E1', '0:D->E1', 64.0), ('F', '1:D->F', 128.0), ('E2', '2:D->E2', 256.0)] successor
        # ['1:A->C', '0:C->D', '0:D->E1'] path
        # ['A', 'B1', 'C', 'B2', 'C', 'D', 'D', 'E1'] visited
        # [('F', '0:E1->F', 512.0)] successor
        # ['1:A->C', '0:C->D', '1:D->F'] path
        # ['A', 'B1', 'C', 'B2', 'C', 'D', 'D', 'E1', 'F'] visited
        # [('G', '0:F->G', 2048.0)] successor
        # ['1:A->C', '0:C->D', '2:D->E2'] path
        # ['A', 'B1', 'C', 'B2', 'C', 'D', 'D', 'E1', 'F', 'E2'] visited
        # [('F', '0:E2->F', 1024.0)] successor
        # ['0:A->B1', '0:B1->C', '0:C->D', '0:D->E1'] path
        # ['A', 'B1', 'C', 'B2', 'C', 'D', 'D', 'E1', 'F', 'E2', 'E1'] visited
        # [('F', '0:E1->F', 512.0)] successor
        # ['0:A->B1', '0:B1->C', '0:C->D', '1:D->F'] path
        # ['A', 'B1', 'C', 'B2', 'C', 'D', 'D', 'E1', 'F', 'E2', 'E1', 'F'] visited
        # [('G', '0:F->G', 2048.0)] successor
        # ['0:A->B1', '0:B1->C', '0:C->D', '2:D->E2'] path
        # ['A', 'B1', 'C', 'B2', 'C', 'D', 'D', 'E1', 'F', 'E2', 'E1', 'F', 'E2'] visited
        # [('F', '0:E2->F', 1024.0)] successor
        # ['1:A->C', '0:C->D', '0:D->E1', '0:E1->F'] path
        # ['A', 'B1', 'C', 'B2', 'C', 'D', 'D', 'E1', 'F', 'E2', 'E1', 'F', 'E2', 'F'] visited
        # [('G', '0:F->G', 2048.0)] successor
        # ['1:A->C', '0:C->D', '1:D->F', '0:F->G'] path
        # ['A', 'B1', 'C', 'B2', 'C', 'D', 'D', 'E1', 'F', 'E2', 'E1', 'F', 'E2', 'F', 'G'] visited

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    visited = []

    if problem.isGoalState(problem.getStartState()):
        return []

    pq = PriorityQueue()  # create a queue
    pq.push((problem.getStartState(), []), 0)  # ((start pos, empty list), 0 cost)

    while pq:
        new_pos, path = pq.pop()

        if new_pos not in visited:
            visited.append(new_pos)
        else:
            continue

        # print(path, 'path')
        # print(visited, 'visited')
        if problem.isGoalState(new_pos):
            return path

        successor = problem.getSuccessors(new_pos)
        # print(successor, 'successor') # [('B', '0:A->B', 1.0), ('C', '1:A->C', 2.0), ('D', '2:A->D', 4.0)]
        # i[2] will be the cost of the successors

        if successor:
            for i in successor:
                if i[0] not in visited:
                    # print(i[2], 'cost') # 1.0 cost, 2.0 cost, 4.0 cost
                    # getCostOfActions to calculate the cost of the paths
                    cost = problem.getCostOfActions(path + [i[1]])
                    # print(cost)
                    pq.push((i[0], path + [i[1]]), cost)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

# f(n) = g(n) + h(n)
# g(n) is the cost of the path from the start node to n
# h(n) is a heuristic that estimates the cost of the cheapest path from n to the target node

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    visited = []

    if problem.isGoalState(problem.getStartState()):
        return []

    apq = PriorityQueue()
    apq.push((problem.getStartState(), []), 0)  # ((start pos, empty list), 0 cost)

    while apq:
        new_pos, path = apq.pop()

        if new_pos not in visited:
            visited.append(new_pos)
        else:
            continue

        if problem.isGoalState(new_pos):
            return path

        successor = problem.getSuccessors(new_pos)

        if successor:
            for i in successor:
                if i[0] not in visited:
                    cost = problem.getCostOfActions(path + [i[1]])
                    # cost + heuristic(i[0], problem) is f(n) = g(n) + h(n)
                    apq.push((i[0], path + [i[1]]), cost + heuristic(i[0], problem))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
