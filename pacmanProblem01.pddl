;; This problem demonstrates that Pacman will avoid enemy Ghosts by taking a
;; longer route to find food. It will then return to a given home position.

(define (problem pacman1)
	(:domain pacman)

    ;; Defines all of the possible locations within the game grid as position objects.
    ;; These are represented by X & Y coordinate values.
	(:objects X1Y1 X1Y2 X1Y3 X1Y4 X1Y5 X2Y1 X2Y3 X2Y5 X3Y1 X3Y3 X3Y5 X4Y1 X4Y3 X4Y5 X5Y1 X5Y2 X5Y3 X5Y4 X5Y5 - position)

    ;; Inititalise the game state with all of the connections between neighbouring positions.
    ;; Also place the Pacman at it's current location (and which has been visited as well).
    ;; Mark also the points where food is located, and whether food is being carried by the Pacman.
	(:init
		(connected X1Y1 X2Y1)
		(connected X1Y1 X1Y2)
		(connected X1Y2 X1Y1)
		(connected X1Y2 X1Y3)
		(connected X1Y3 X1Y2)
		(connected X1Y3 X1Y4)
		(connected X1Y3 X2Y3)
		(connected X1Y4 X1Y3)
		(connected X1Y4 X1Y5)
		(connected X1Y5 X1Y4)
		(connected X1Y5 X2Y5)
		(connected X2Y1 X1Y1)
		(connected X2Y1 X3Y1)
		(connected X2Y3 X1Y3)
		(connected X2Y3 X3Y3)
		(connected X2Y5 X1Y5)
		(connected X2Y5 X3Y5)
		(connected X3Y1 X2Y1)
		(connected X3Y1 X4Y1)
		(connected X3Y3 X2Y3)
		(connected X3Y3 X4Y3)
		(connected X3Y5 X2Y5)
		(connected X3Y5 X4Y5)
		(connected X4Y1 X3Y1)
		(connected X4Y1 X5Y1)
		(connected X4Y3 X3Y3)
		(connected X4Y3 X5Y3)
		(connected X4Y5 X3Y5)
		(connected X4Y5 X5Y5)
		(connected X5Y1 X4Y1)
		(connected X5Y1 X5Y2)
		(connected X5Y2 X5Y1)
		(connected X5Y2 X5Y3)
		(connected X5Y3 X4Y3)
		(connected X5Y3 X5Y2)
		(connected X5Y3 X5Y4)
		(connected X5Y4 X5Y3)
		(connected X5Y4 X5Y5)
		(connected X5Y5 X4Y5)
		(connected X5Y5 X5Y4)
        (hasFood X1Y3)
        (hasFood X2Y3)
        (hasFood X3Y3)
        (hasFood X4Y3)
        (hasFood X5Y3)
		(at X1Y1)
        (visited X1Y1)
	)

    ;; To achieve its goal, a Pacman is tasked with visiting a location where food
    ;; is known to be, and bringing it back to one or more home locations. While this
    ;; occurs it must also avoid positions that have enemy ghosts as well as
    ;; positions next to them. The game itself will decide if Ghosts are a threat
    ;; or not by monitoring if Pacman is SuperPacman. If it is, then the ghosts
    ;; will not be added to the problem definition.
    (:goal
		(and
			(not (visited X1Y3))
			(not (visited X1Y2))
			(not (visited X1Y4))
            (not (visited X2Y3))
			(carryingFood)
			(at X1Y1)
		)
	)
)
