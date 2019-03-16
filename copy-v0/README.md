# TapeAlogirhtm - Copy-v0

Alphabet `A`..`E` are stored on a tape

```
Total length of input instance: 4, step: 4
==========================================
Observation Tape    :   AAEA
Output Tape         :
Targets             :   AAEA
```

A head is arrowing a cell on the (`observation`) tape.
Your action is move the head to left/right AND write an alphabet to another (`output`) tape or not write (moving always be specified).
A state is an alphabet arrowed by the head.

`Copy-v0`'s goal is copy the alphabets on the observation tape to the output tape..

```
Total length of input instance: 4, step: 4
==========================================
Observation Tape    :   AAEA
Output Tape         :   AAEA
Targets             :   AAEA
```

## Q-learning

```bash
python ./q.py
```

The space of status and action is discrete.
Status space is `{A,B,C,D,E,NUL}` (NUL is out of tape).
Action space is `{Left, Right} x {Write, NotWrite} x {A,B,C,D,E}`.
Simply, I take a space `6 x 2 x 2 x 5` as a Q-table.
Each cell has a Q-value, goodness.
