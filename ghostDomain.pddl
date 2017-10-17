(define (domain ghost)
    (:requirements :typing)
    (:types position)

    ;; The predicates represent a set of connected grid locations that can be
    ;; moved between and visited.
    (:predicates
        (move ?from ?to - position)
        (at ?pos - position)
        (visited ?pos - position)
        (connected ?from ?to - position)
    )

    ;; A Ghost's only action available is to move to a connected position.
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
                (not (at ?from))
                (at ?to)
                (visited ?to)
            )
    )
)
