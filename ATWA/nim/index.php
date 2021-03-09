<?php

require_once(__DIR__ . '/template.php');


function tpl_match($is_linked = false, $link_url = null, $classes = "")
{
    if ($is_linked)
        echo '<a href="?' . $link_url . '" class="match">';
    echo '<img src="style/match.png" class="match ' . $classes . '">';
    if ($is_linked)
        echo '</a>';
}

function best_move($rest) {
    $a = 0;
    $tmp = 0;
    for($k = 0; $a < $rest; $k++) {
        $tmp = $a;
        $a = 3 * $k + $k + 1;
    }

    if ($rest - $tmp > 3) {
        return 0;
    } else {
        return $rest - $tmp;
    }
}

function render_game() {
    if (isset($_GET["matches"])) {
        $matches_num = $_GET["matches"];
    } elseif (isset($_GET["initial"])) {
        $matches_num = $_GET["initial"];
    } else {
        $matches_num = 0;
    }

    $taken_matches = 0;
    $won = 0;

    if (isset($_GET["matches"])) {
        if ($matches_num == 0) {
            $won = 2;
        } elseif ($matches_num == 1) {
            $taken_matches = 1;
            $won = 1;
        } elseif (best_move($matches_num)) { // winning move
            $taken_matches = best_move($matches_num);
        } else {
            srand($_GET["seed"]);
            $taken_matches = rand(1, 3);
        }
    }

    echo '<div class="center">';
    for ($i = 1; $i <= $matches_num - $taken_matches; $i++) {
        if ($i <= 3) {
            $next_matches = $matches_num - $i - $taken_matches;
            $now = new DateTime();
            $ts = $now->getTimestamp();
            tpl_match(true,
                "initial=" . $_GET["initial"] . "&" . "matches=" . $next_matches . "&seed=" . $ts
            );
        } else {
            tpl_match();
        }
    }

    for ($i = 0; $i < $taken_matches; $i++) {
        tpl_match(false, "", "taken");
    }
    echo '</div>';

    if ($won != 0 && isset($_GET["matches"])) {
        echo '<p>There are no matches left...</p>';
        echo '<p>Game over. The winner is <strong>' . ($won == 1 ? "player" : "server") . '</strong>!<br><a href="?">Play Again</a></p>';
    }
}

function main() {
    if (
        (isset($_GET["initial"]) && !is_numeric($_GET["initial"]) && !empty($_GET["initial"]) ) ||
        (isset($_GET["seed"]) && !is_numeric($_GET["seed"]) && !empty($_GET["seed"])) ||
        (isset($_GET["matches"]) && !is_numeric($_GET["matches"]) && !empty($_GET["matches"]))
    ) {
        http_response_code(400);
    } else {
        render_game();
    }
}
