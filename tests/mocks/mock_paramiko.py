def AutoAddPolicy():
    return True


class SSHClient:
    def __init__(self):
        self.logged_in = False

        good_input_1 = """-q \
SELECT firstseen, lastseen, callsign, icao24, estdepartureairport, estarrivalairport, day
    FROM flights_data4 
    WHERE estdepartureairport = 'KBTR' 
    AND estarrivalairport = 'KDFW'
    AND day >= 1685577600
    AND day <= 1685836800
    ORDER BY firstseen;"""

        good_stdout_1 = b""

        good_stderr_1 = b"""
WARNINGS: Disk I/O error: Error reading from HDFS file: hdfs://nameservice1/user/opensky/tables_v4/flights/day=1685836800/part-r-00169-ddd97750-7950-42c9-b236-4a9c7e9106ae.snappy.parquet
Error(255): Unknown error 255
Root cause: BlockMissingException: Could not obtain block: BP-2086186090-192.168.6.170-1416410368441:blk_1716518662_875005380 file=/user/opensky/tables_v4/flights/day=1685836800/part-r-00169-ddd97750-7950-42c9-b236-4a9c7e9106ae.snappy.parquet"""

        good_input_2 = """-q \
SELECT firstseen, lastseen, callsign, icao24, estdepartureairport, estarrivalairport, day
    FROM flights_data4 
    WHERE estdepartureairport = 'KBTR' 
    AND estarrivalairport = 'KDFW'
    AND day >= 1685577600
    AND day <= 1685836800
    AND day != 1685836800
ORDER BY firstseen;"""

        good_stdout_2 = b"""\
+------------+------------+----------+--------+---------------------+-------------------+------------+
| firstseen  | lastseen   | callsign | icao24 | estdepartureairport | estarrivalairport | day        |
+------------+------------+----------+--------+---------------------+-------------------+------------+
| 1685580348 | 1685583833 | ENY3479  | a1cd4e | KBTR                | KDFW              | 1685577600 |
| 1685635052 | 1685638701 | ENY3575  | a1c229 | KBTR                | KDFW              | 1685577600 |
| 1685651296 | 1685654994 | ENY3431  | a2d6a2 | KBTR                | KDFW              | 1685577600 |
| 1685658580 | 1685662333 | SKW4906  | aa5d23 | KBTR                | KDFW              | 1685577600 |
| 1685667930 | 1685671930 | SKW3021  | a99686 | KBTR                | KDFW              | 1685664000 |
| 1685704862 | 1685708876 | ENY3664  | a24782 | KBTR                | KDFW              | 1685664000 |
| 1685720872 | 1685725434 | ENY3704  | a24782 | KBTR                | KDFW              | 1685664000 |
| 1685733665 | 1685737318 | ENY3431  | a2ddac | KBTR                | KDFW              | 1685664000 |
| 1685745530 | 1685749472 | SKW4906  | aa0a6e | KBTR                | KDFW              | 1685664000 |
| 1685753450 | 1685757323 | SKW3021  | aa11dc | KBTR                | KDFW              | 1685750400 |
| 1685790696 | 1685794703 | ENY3664  | a24782 | KBTR                | KDFW              | 1685750400 |
| 1685806879 | 1685810528 | ENY3704  | a214de | KBTR                | KDFW              | 1685750400 |
| 1685821356 | 1685825333 | JIA5074  | a7c878 | KBTR                | KDFW              | 1685750400 |
+------------+------------+----------+--------+---------------------+-------------------+------------+"""

        good_stderr_2 = b""

        # This variable should hold all outputs for a given input
        self.stdout = {
            good_input_1: MockStdout(good_stdout_1),
            good_input_2: MockStdout(good_stdout_2),
        }

        self.stderr = {
            good_input_1: MockStdout(good_stderr_1),
            good_input_2: MockStdout(good_stderr_2),
        }

    def connect(self, hostname, port="0", username="admin", password="password"):
        mock_correct_username = "admin"
        mock_correct_password = "password"
        mock_correct_port = "0"
        mock_correct_hostname = "ssh.mock.fake"
        if mock_correct_username != username:
            raise ValueError("Invalid Username")
        if mock_correct_password != password:
            raise ValueError("Invalid Password")
        if mock_correct_port != port:
            raise ValueError("Invalid Port")
        if mock_correct_hostname != hostname:
            raise ValueError("Invalid Hostname")

        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.logged_in = True

    def set_missing_host_key_policy(self, policy):
        if policy:
            pass
        else:
            raise ValueError(f"Policy not valid")

    def exec_command(self, cmd):
        if self.logged_in:
            if cmd in self.stdout:
                return MockStdout(cmd), self.stdout[cmd], self.stderr[cmd]

            else:
                raise ValueError(f"Invalid Command: {cmd}")
        else:
            raise ValueError(f"Not Logged In")

    def close(self):
        self.logged_in = False


class MockStdout:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data
