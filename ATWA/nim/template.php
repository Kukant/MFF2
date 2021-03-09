<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>NIM</title>
    <link rel="stylesheet" href="style/main.css" type="text/css">
</head>


<body>
<h1>NIM</h1>
<p>Player and computer take 1-3 matches in turns. Whoever takes the last match, looses.</p>



<!-- initial form (start a game) -->
<?php
    if (!isset($_GET["initial"]))
        echo '<form action="?" method="GET">
                <table>
                    <tr>
                        <td><label>Matches:</label></td>
                        <td class="center"><input type="number" min="2" max="50" value="20" name="initial"></td>
                    </tr>
                    <tr>
                        <td colspan="2" class="center"><button type="submit">Start</button></td>
                    </tr>
                </table>
            </form>';

    main();
?>

</body>
</html>
