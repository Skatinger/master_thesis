

## Question Answering Models
Probably not very useful, as they are tuned to find the answer in the given text.
Due to the fact that in the wikipedia dataset the name never occurs, the model never answers
with the name of the person, but always with a placeholder, e.g. "<mask>" or "the 16th president
of the united states". If only a single sentence gives the name of the placeholder, the answer
might be correct.
#### Wiki vs Court Decisions
Question answering works poorly on the wiki dataset, but for the court decision reidentifaction
it might work better, as it is possible to feed the model related news articles beforehand.
An example with DJ Bobo and `deepset/roberta-base-squad2`:

**snippet from the court decision:**
> Zur Begründung wurde im Wesentlichen ausgeführt, dass es sich beim
> Kläger um den Weltstar DJ Bobo handle. Seine zahlreichen Fernsehauftritte,
> Konzert-Touren, Sponsorverbindungen sowie die prämierten Mega-Hits würden
> das unwahrscheinliche Interesse an dieser Person zeigen.  <mask> alias DJ Bobo sei bei der gesamten schweizerischen
> Bevölkerung berühmt. Der Kläger sei
> Inhaber der Einzelfirma G. mit Sitz in R. . Der Zweck dieser Firma sei
> die Vermittlung von Künstlern und Artisten; die Organisation von Anlässen und
> Tourneen; Management und Konzertberatung von Diskotheken und Künstlern;
> Kinoproduktionen; Handel und Verkauf von Tonträgern sowie Zubehör von
> Discjockeys.

**Newsarticle about the ruling:**
> DJ BoBo kann die Internetadresse «www.djbobo.de» für sich beanspruchen. Das Bundesgericht hat ein Urteil des
> Nidwaldner Kantonsgerichts vom letzten November bestätigt. Das Kantonsgericht hatte im November 2001 entschieden, die
> Verwendung des Domain-Namens «www.djbobo.de» durch die Firma Second Label GmbH verletze die Namens- und Markenrechte
> von Renè Baumann, besser bekannt als DJ BoBo.

**Precise question:**
> What is the real name of <mask>? hint: its not DJ BoBo nor DJ Bobo

**Answer (correct):**
> Renè Baumann

First and second prediction were *DJ BoBo* and *DJ Bobo*, but the third was the correct name.
In cases where aliases are not as explicitly used, the first prediction might already be the correct one.

# Text Generation Models
With good prompts and adequate zero shot training, text generation models are able to predict persons on wikipedia
directly from their input parameter knowledge. Interesting here is that these models could combine knowledge from
their training with additional inputs, for example a quick keyword search over all news articles might serve articles
which are relevant and could give a hint about the person, allowing the models to more accuractly or more often
reidentify the person. The peformance on wikipedia seems to correlate to the importance of the wiki person.

- Number of masks does not seem to correlate with accuracy of result, at least not for the very large models
bloomz-176 and gpt-3.5.turbo.