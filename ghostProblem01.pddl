(define (problem ghost1)
	(:domain ghost)

    ;; Defines all of the possible locations within the game grid as position objects.
    ;; These are represented by X & Y coordinate values.
	(:objects X1Y1 X1Y2 X1Y3 X1Y4 X1Y5 X2Y1 X2Y3 X2Y5 X3Y1 X3Y3 X3Y5 X4Y1 X4Y3 X4Y5 X5Y1 X5Y2 X5Y3 X5Y4 X5Y5 - position)

    ;; Inititalise the game state with all of the connections between neighbouring positions.
    ;; Also place the Ghost at it's current location (and which has been visited as well).
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
		(at X1Y1)
        (visited X1Y1)
	)

    ;; The game state will determine which of the enemy agents are considered targets
    ;; for the Ghost. A Pacman will need to be visible, in the Ghost's defensive side
    ;; and not currently a SuperPacman. The PDDL model does not take this into account
    ;; but instead is provided a list of positions it needs to visit, representing
    ;; these enemy agents to target.
    (:goal
        (and
            (visited X3Y3)
            (visited X5Y1)
        )
    )
)
