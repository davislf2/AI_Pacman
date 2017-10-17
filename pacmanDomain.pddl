(define (domain pacman)
    (:requirements :typing)
    (:types position)

    ;; The predicates represent a set of connected grid locations that can be
    ;; moved between and visited, in addition to facts that keep track of
    ;; the food located in the grid, and carried by Pacman.
    (:predicates
        (move ?from ?to - position)
        (at ?pos - position)
        (visited ?pos - position)
        (connected ?from ?to - position)
        (eat ?pos - position)
        (hasFood ?pos - position)
        (carryingFood)
    )

    ;; A Pacman is able to move to a connected position.
    (:action move
        :parameters
            (?from ?to - position)
        :precondition
            (and
                (at ?from)
                (connected ?from ?to)
            )
        :effect
            (and
                (at ?to)
                (not (at ?from))
                (visited ?to)
            )
    )

    ;; While eating occurs instantaneously in the main game, we can separate
    ;; this into a second action for planning purposes as a simplification.
    (:action eat
        :parameters
            (?pos - position)
        :precondition
            (and
                (at ?pos)
                (hasFood ?pos)
            )
        :effect
            (and
                (carryingFood)
                (not (hasFood ?pos))
            )
    )
)
