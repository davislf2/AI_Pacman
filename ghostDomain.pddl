(define (domain ghost)
    (:requirements :typing)
    (:types position)

    ;; Define the facts in the problem
    ;; "?" denotes a variable, "-" a type
    (:predicates
        (move ?from ?to - position)
        (at ?pos - position)
        (connected ?pos1 ?pos2 - position)
    )

    ;; Define the action(s)
    (:action move
        :parameters
            (?from ?to - position)
        :precondition
            (and (at ?from)
            (connected ?from ?to))
        :effect
            (and (at ?to)
            (not (at ?from)))
    )
)
