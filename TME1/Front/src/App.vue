<template>
  <div id="app">
    <div id="app">
      <div class="home">
        <img src="/img/book.png" alt="" />
        <div class="title">Gutenberg project</div>

        <div class="links">
          <a class="link" href="" target="_blank" rel="noopener">
            Find books
          </a>
          <a class="link" href="" target="_blank" rel="noopener">
            The Gutenberg project
          </a>
          <a class="link" href="" target="_blank" rel="noopener">
            DAAR
          </a>
          <a class="link" href="" target="_blank" rel="noopener">
            This app repository
          </a>
        </div>
      </div>
    </div>
  </div>
</template>

<style>

html, body, #app {
  height: 100%;
  margin: 0;
  background: #efefef;
}

.home {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 40px;
  background: #efefef;
}

.home > img {
  width: min(520px, 80vw);
  height: auto;
  display: block;
  filter: grayscale(0);
}

.title {
  font-weight: 700;
  font-size: clamp(28px, 5vw, 40px);
  color: #0b0b0b;
  text-align: center;
}

.links {
  width: min(700px, 92vw);
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 22px;
}

.link {
  display: grid;
  place-items: center;
  height: 110px;
  border-radius: 14px;
  background: #ffffff;
  color: #111;
  text-decoration: none;
  border: 1px solid #e5e5e5;
}


@media (max-width: 680px) {
  .links { grid-template-columns: 1fr; }
  .link { height: 84px; }
}
</style>
