
This project uses **MkDocs** to build and publish documentation.
Follow the steps below to update and deploy the docs.

------------------------------------------------------------------------

## How to locally build and push a documentation update

1.  Clone the repository (if not already done):

    ``` bash
    git clone https://github.com/yourusername/yourrepo.git
    cd yourrepo
    ```

2.  Install MkDocs:

    ``` bash
    pip install mkdocs mkdocs-material
    ```

3.  Edit or update the Markdown files inside the `docs/` folder.

4.  Preview changes locally:

    ``` bash
    mkdocs serve
    ```

    Open your browser at <http://127.0.0.1:8000> to see the live
    preview.

5.  Build the documentation site:

    ``` bash
    mkdocs build
    ```

6.  Deploy (GitHub Pages example):

    ``` bash
    mkdocs gh-deploy
    ```

------------------------------------------------------------------------

## Commands

-   `mkdocs serve` → Start the live-reloading docs server\
-   `mkdocs build` → Build the documentation site\
-   `mkdocs gh-deploy` → Deploy to GitHub Pages
