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
2024-01-17 03:18:35 Building Witnesses for block 1461085. Progress=0.420616
2024-01-17 03:18:43 Writing witness building progress to the disk.
2024-01-17 03:20:25 SetBestChain(): SetBestChain was successful
2024-01-17 03:20:38 Building Witnesses for block 1461112. Progress=0.420623
2024-01-17 03:21:38 Building Witnesses for block 1461452. Progress=0.420711
2024-01-17 03:22:38 Building Witnesses for block 1461790. Progress=0.420805
2024-01-17 03:23:38 Building Witnesses for block 1462083. Progress=0.420901
2024-01-17 03:24:38 Building Witnesses for block 1462386. Progress=0.420982
2024-01-17 03:25:38 Building Witnesses for block 1462725. Progress=0.421076
2024-01-17 03:26:39 Building Witnesses for block 1463040. Progress=0.421171
2024-01-17 03:27:41 Building Witnesses for block 1463316. Progress=0.421250
2024-01-17 03:28:42 Building Witnesses for block 1463659. Progress=0.421345
2024-01-17 03:28:45 Writing witness building progress to the disk.
2024-01-17 03:30:48 SetBestChain(): SetBestChain was successful
2024-01-17 03:30:55 Building Witnesses for block 1463670. Progress=0.421346
2024-01-17 03:31:56 Building Witnesses for block 1464023. Progress=0.421442
2024-01-17 03:32:56 Building Witnesses for block 1464397. Progress=0.421546

```

If message is Building Witness Cache, incriment time by 1 minute. If progress is avalaible, record for report and print progress