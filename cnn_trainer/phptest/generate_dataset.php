<?php

require_once __DIR__.'/vendor/autoload.php';

use Gregwar\Captcha\CaptchaBuilder;
use Gregwar\Captcha\PhraseBuilder;

// Will build phrases of 3 characters
// $phraseBuilder = new PhraseBuilder(4);

// Will build phrases of 5 characters, only digits
$phraseBuilder = new PhraseBuilder(5, '2456789бвrджклмнпрст'); # '2456789бвгдежклмнпрст'

// Pass it as first argument of CaptchaBuilder, passing it the phrase
// builder
$captcha = new CaptchaBuilder(null, $phraseBuilder);


for ($x = 0; $x <= 40000; $x++) {
  echo "The number is: $x <br>";
  $phrase = $phraseBuilder->build();
  $captcha->setPhrase($phrase);
  $captcha
    ->build(200, 60)
    ->save(__DIR__.'/gen_train/'.$phrase.'.jpg');
}
?>
