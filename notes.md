## TODO:

### Zapping Wallet Transactions
Track zapping wallet transaction work

Example debug message while zapping
```
2024-01-09 02:55:06 init message: Zapping all transactions from wallet...
```

Example `getinfo` response while zapping:
```
error code: -28
error message:
Zapping all transactions from wallet...
```

Maybe create a `zapping = False` variable. Check if zapping is True, if not, put the start timestamp in zapping_starttime. Will need a completion trigger 

---

### Building Witnes Cache

improve building witness cache. Example debug log message

```
2024-01-17 03:22:38 Building Witnesses for block 1461790. Progress=0.420805
```

If message is Building Witness Cache, incriment time by 1 minute. If progress is avalaible, record for report and print progress